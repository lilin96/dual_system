from TCP.model import TCP
import torch


class TCPVLM(TCP):

    def __init__(self, **kwargs):
        super(TCPVLM, self).__init__(**kwargs)

    def forward(self, img, state, target_point):
        feature_emb, cnn_feature = self.perception(img)
        outputs = {}
        outputs['pred_speed'] = self.speed_branch(feature_emb)
        measurement_feature = self.measurements(state)

        j_traj = self.join_traj(torch.cat([feature_emb, measurement_feature], 1))
        outputs['pred_value_traj'] = self.value_branch_traj(j_traj)
        outputs['pred_features_traj'] = j_traj
        z = j_traj
        output_wp = list()
        traj_hidden_state = list()

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).type_as(z)

        # autoregressive generation of output waypoints
        for _ in range(self.config.pred_len):
            x_in = torch.cat([x, target_point], dim=1)
            z = self.decoder_traj(x_in, z)
            traj_hidden_state.append(z)
            dx = self.output_traj(z)
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)
        outputs['pred_wp'] = pred_wp

        traj_hidden_state = torch.stack(traj_hidden_state, dim=1)
        init_att = self.init_att(measurement_feature).view(-1, 1, 8, 29)
        feature_emb = torch.sum(cnn_feature * init_att, dim=(2, 3))
        j_ctrl = self.join_ctrl(torch.cat([feature_emb, measurement_feature], 1))
        outputs['pred_value_ctrl'] = self.value_branch_ctrl(j_ctrl)
        outputs['pred_features_ctrl'] = j_ctrl
        policy = self.policy_head(j_ctrl)
        outputs['mu_branches'] = self.dist_mu(policy)
        outputs['sigma_branches'] = self.dist_sigma(policy)

        x = j_ctrl
        mu = outputs['mu_branches']
        sigma = outputs['sigma_branches']
        future_feature, future_mu, future_sigma = [], [], []

        # initial hidden variable to GRU
        h = torch.zeros(size=(x.shape[0], 256), dtype=x.dtype).type_as(x)

        for _ in range(self.config.pred_len):
            x_in = torch.cat([x, mu, sigma], dim=1)
            h = self.decoder_ctrl(x_in, h)
            wp_att = self.wp_att(torch.cat([h, traj_hidden_state[:, _]], 1)).view(-1, 1, 8, 29)
            new_feature_emb = torch.sum(cnn_feature * wp_att, dim=(2, 3))
            merged_feature = self.merge(torch.cat([h, new_feature_emb], 1))
            dx = self.output_ctrl(merged_feature)
            x = dx + x

            policy = self.policy_head(x)
            mu = self.dist_mu(policy)
            sigma = self.dist_sigma(policy)
            future_feature.append(x)
            future_mu.append(mu)
            future_sigma.append(sigma)

        outputs['future_feature'] = future_feature
        outputs['future_mu'] = future_mu
        outputs['future_sigma'] = future_sigma
        return outputs
