import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagRefineMLP(nn.Module):
    def __init__(
        self,
        main_hidden=32,
        aux_hidden=16,
        aux_out_dim=8,
        fusion_hidden=32,
        drop_main=0.2,
        drop_aux=0.1,
        drop_fusion=0.1,
        threshold=4.0,      # 4分预警
        gate_scale=1.5,
        aux_weight=0.05,     # 辅助头分支降权
        mel_index=2         # 按你当前代码习惯，MEL_INDEX=2；如果类别顺序不同，这里要改
    ):
        super(DiagRefineMLP, self).__init__()

        self.threshold = threshold
        self.gate_scale = gate_scale
        self.aux_weight = aux_weight
        self.mel_index = mel_index

        # 主分支输入:
        # diag_prob(5) + seven_score_norm(1) + rule_gate(1) = 7
        main_in_dim = 5 + 1 + 1
        self.main_mlp = nn.Sequential(
            nn.Linear(main_in_dim, main_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_main),

            nn.Linear(main_hidden, main_hidden),
            nn.ReLU(inplace=True)
        )

        # 辅分支输入:
        # pn(3) + bwv(2) + vs(3) + pig(3) + str(3) + dag(3) + rs(2) = 19
        aux_in_dim = 3 + 2 + 3 + 3 + 3 + 3 + 2
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_in_dim, aux_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_aux),

            nn.Linear(aux_hidden, aux_out_dim),
            nn.ReLU(inplace=True)
        )

        # 融合层:
        # 这里只输出 1 维，只修正 MEL 那一类
        fusion_in_dim = main_hidden + aux_out_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_fusion),

            nn.Linear(fusion_hidden, 1)   # 只输出 delta_mel
        )

        # -----------------------------：
        # 在 __init__ 末尾，把 fusion_mlp 最后一层权重和偏置清零
        # 这样初始时 delta_mel = 0，模型一开始等价于原始 diag 头
        # -----------------------------
        nn.init.constant_(self.fusion_mlp[-1].weight, 0.0)
        nn.init.constant_(self.fusion_mlp[-1].bias, 0.0)

    def compute_seven_point_score(
        self,
        pn_prob,
        bwv_prob,
        vs_prob,
        pig_prob,
        str_prob,
        dag_prob,
        rs_prob
    ):

        score = (
            2.0 * pn_prob[:, 2] +
            2.0 * bwv_prob[:, 1] +
            2.0 * vs_prob[:, 2] +
            1.0 * str_prob[:, 2] +
            1.0 * dag_prob[:, 2] +
            1.0 * pig_prob[:, 2] +
            1.0 * rs_prob[:, 1]
        )
        return score

    def forward(
        self,
        diag_logits,
        pn_logits,
        bwv_logits,
        vs_logits,
        pig_logits,
        str_logits,
        dag_logits,
        rs_logits
    ):
        # ---------- 1) 各头转概率 ----------
        diag_prob = F.softmax(diag_logits, dim=1)   # [B,5]
        pn_prob   = F.softmax(pn_logits, dim=1)     # [B,3]
        bwv_prob  = F.softmax(bwv_logits, dim=1)    # [B,2]
        vs_prob   = F.softmax(vs_logits, dim=1)     # [B,3]
        pig_prob  = F.softmax(pig_logits, dim=1)    # [B,3]
        str_prob  = F.softmax(str_logits, dim=1)    # [B,3]
        dag_prob  = F.softmax(dag_logits, dim=1)    # [B,3]
        rs_prob   = F.softmax(rs_logits, dim=1)     # [B,2]

        # ---------- 2) 七点评分 ----------
        seven_score = self.compute_seven_point_score(
            pn_prob, bwv_prob, vs_prob, pig_prob, str_prob, dag_prob, rs_prob
        )   # [B]

        seven_score_norm = (seven_score / 10.0).unsqueeze(1)   # [B,1]

        # 3分预警门控
        rule_gate = torch.sigmoid(
            self.gate_scale * (seven_score - self.threshold)
        ).unsqueeze(1)   # [B,1]

        # ---------- 3) 主分支：diag_prob + 七点规则 ----------
        main_input = torch.cat([
            diag_prob,          # [B,5]
            seven_score_norm,   # [B,1]
            rule_gate           # [B,1]
        ], dim=1)               # [B,7]

        main_feat = self.main_mlp(main_input)       # [B, main_hidden]

        # ---------- 4) 辅分支：7个辅助头（弱参考） ----------
        aux_input = torch.cat([
            pn_prob,            # [B,3]
            bwv_prob,           # [B,2]
            vs_prob,            # [B,3]
            pig_prob,           # [B,3]
            str_prob,           # [B,3]
            dag_prob,           # [B,3]
            rs_prob             # [B,2]
        ], dim=1)               # [B,19]

        aux_feat = self.aux_mlp(aux_input)          # [B, aux_out_dim]
        aux_feat = self.aux_weight * aux_feat       # 显式降权

        # ---------- 5) 融合，只输出 melanoma 修正量 ----------
        fusion_input = torch.cat([main_feat, aux_feat], dim=1)
        raw_delta_mel = self.fusion_mlp(fusion_input)  # [B,1]
        delta_mel = 0.5 * torch.tanh(raw_delta_mel)
        # ---------- 6) 只修正 MEL 那一维 ----------
        final_diag = diag_logits.clone()
        final_diag[:, self.mel_index:self.mel_index + 1] = \
            final_diag[:, self.mel_index:self.mel_index + 1] + delta_mel

        # ---------- 7) 调试信息 ----------
        aux = {
            "diag_prob": diag_prob,                     # [B,5]
            "seven_score": seven_score,                 # [B]
            "seven_score_norm": seven_score_norm,       # [B,1]
            "rule_gate": rule_gate,                     # [B,1]
            "main_feat": main_feat,                     # [B,main_hidden]
            "aux_feat": aux_feat,                       # [B,aux_out_dim]
            "delta_mel": delta_mel                      # [B,1]
        }

        return final_diag, aux