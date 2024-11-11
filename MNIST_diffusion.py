import torch
from model import MNISTDiffusion


class MNIST_Model():
    def __init__(self, pathtoModel: str, timesteps: int = 1000):
        # Initialize the parent class with required parameters
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.timesteps = timesteps
        self.Model = MNISTDiffusion(timesteps=self.timesteps,
                                    image_size=28,
                                    in_channels=1,
                                    base_dim=64,
                                    dim_mults=[2, 4]).to(self.device)

        self.sqrt_one_minus_alphas_cumprod = self._compute_sqrt_one_minus_alphas_cumprod()
        self.load_model_state(pathtoModel)

    def _compute_sqrt_one_minus_alphas_cumprod(self):
        # This function computes the sqrt(1 - cumulative product of alphas)
        alphas = torch.linspace(0.01, 1.0, self.timesteps).to(
            self.device)  # Replace with actual alpha values
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return torch.sqrt(1 - alphas_cumprod)

    def load_model_state(self, path: str):
        """Load the model state from a saved file."""
        try:

            # Load the checkpoint
            # added for compatibility
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=False)
            # Load the model state dict
            self.Model.load_state_dict(checkpoint['model'])
            self.Model.eval()  # Set the model to evaluation mode
            print(f"Model loaded successfully")
        except Exception as e:
            print(f"Failed to load model from {path}: {e}")

    def get_score(self, x_t, t: float = 0):
        """Get the score function for the input x."""
        assert 0 <= t <= 1, "Time t must be between 0 and 1."
        # Ensure x_t is on the correct device
        x_t = x_t.requires_grad_(True).to(self.device)

        # Convert the time to an integer within the range of timesteps
        t = int(t * self.timesteps - 1)
        # Prepare the time tensor
        t = torch.full((x_t.shape[0],), t,
                       device=self.device, dtype=torch.long)

        # Generate noise with the same shape as x_t
        noise = torch.randn_like(x_t)

        # Perform reverse diffusion to get the predicted noise
        pred_noise = self.Model._reverse_diffusion_with_grad(x_t, t, noise)

        # Ensure pred_noise has requires_grad=True (if needed)
        pred_noise = pred_noise.clone().detach().requires_grad_(True)

        # Compute score by scaling the predicted noise
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(
            -1, t).reshape(x_t.shape[0], 1, 1, 1)

        score = pred_noise / sqrt_one_minus_alpha_cumprod_t

        return score.mean()  # Return the mean score across the batch


if __name__ == "__main__":
    print("Loaded")
