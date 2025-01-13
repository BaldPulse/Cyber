from cyber.models.action.imitation.backbones.dit import DiTNoiseNet
import torch

from tests.utils import reseed_everything


class TestDiTNoiseNet:
    @classmethod
    def setup_class(cls):

        reseed_everything()

        cls.action_dim = 2
        cls.chunk_size = 2
        cls.hidden_size = 512
        cls.num_blocks = 6

        cls.model = DiTNoiseNet(ac_dim=cls.action_dim,
                                ac_chunk=cls.chunk_size,
                                hidden_dim=cls.hidden_size,
                                num_blocks=cls.num_blocks)
        
        reseed_everything()
        cls.input = {
            "noise_actions": torch.randn(3, 2, 2), # (batch_size, ac_chunk, ac_dim)
            "time": torch.randn(3), # (batch_size, 1)
            "obs_enc": torch.randn(3, 42, 512) # (batch_size, num_tokens, hidden_dim)
        }
        
    @classmethod
    def teardown_class(cls):
        del cls.model

    def test_forward(self):
        self.model.train()
        with torch.no_grad():
            output = self.model(**self.input)
        assert output[1].shape == (3, 2, 2), f"output shape is {output.shape}"
        # load prediction from file
        expected_output = torch.load("tests/fixtures/tensors/dit_noisenet_prediction.pth")
        for i in range(len(expected_output[0])):
            assert torch.allclose(output[0][i], expected_output[0][i], atol=1e-5), f"output is not equal"

        assert torch.allclose(output[1], expected_output[1]), f"output is not equal"