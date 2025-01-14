from cyber.models.action.diffusion.backbones import ConditionalUnet1D
import torch

from tests.utils import reseed_everything


class TestConditionalUnet1D:
    @classmethod
    def setup_class(cls):

        reseed_everything()

        action_dim = 6
        global_condition_dim = (512 + 6) * 4
        local_condition_dim = 256

        cls.model = ConditionalUnet1D(input_dim=action_dim, 
                                      global_cond_dim=global_condition_dim, 
                                      local_cond_dim=local_condition_dim)
        
        reseed_everything()
        cls.input = {
            "noise_actions": torch.randn(3, 4, 6), # (batch_size, ac_chunk, ac_dim)
            "time_step": torch.randn(3), # (batch_size,)
            "condition": torch.randn(3, global_condition_dim), # (batch_size, act_horizon, global_condition_dim)
            "local_cond": torch.randn(3, 4, local_condition_dim) # (batch_size, act_horizon, local_condition_dim)
        }
        
    @classmethod
    def teardown_class(cls):
        del cls.model

    def test_forward(self):
        self.model.train()
        with torch.no_grad():
            output = self.model(**self.input)
        assert output.shape == (3, 4, 6), f"output shape is {output.shape}"
        # load prediction from file
        expected_output = torch.load("tests/fixtures/tensors/conditional_unet1d_prediction.pth")
        assert torch.allclose(output, expected_output, atol=1e-5), f"output is not equal"
