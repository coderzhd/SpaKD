import torch

from spakd.model import SpaKDConfig, SpaKDModule


def test_spakd_forward_and_loss():
    config = SpaKDConfig(
        sc_dim=5,
        st_dim=7,
        latent_dim=16,
        student_hidden_dims=(32,),
        teacher_hidden_dims=(32,),
        num_bases=4,
        projection_dim=8,
        projection_hidden_dim=16,
    )
    module = SpaKDModule(config)
    batch_size = 6
    module.set_input(
        {
            "scx": torch.rand(batch_size, config.sc_dim),
            "stx": torch.rand(batch_size, config.st_dim),
            "gene_group": torch.tensor([0, 1, 2, 0, 1, 2]),
            "n_original": batch_size,
        },
        train=True,
    )
    out = module.forward()
    assert out["student_pred"].shape == (batch_size, config.st_dim)
    loss = module.compute_loss()
    assert torch.isfinite(loss)
