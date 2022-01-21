from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    """Base class for implemting different losses"""

    def __init__(self, name, **kwargs) -> None:
        super().__init__()
        self.name = name

    def disc_loss(
        self,
        fake_preds: torch.Tensor,
        real_preds: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """[summary]

        Parameters
        ----------
        fake_preds : torch.Tensor
            output of the discriminator for fake samples, shape = [batch_size, ]
        real_preds : torch.Tensor
            output of the discriminator for real samples, shape = [batch_size, ]

        Returns
        -------
        torch.Tensor
            return the computed loss for the discriminator forward step

        Raises
        ------
        NotImplementedError
            child class should implement this method
        """
        raise NotImplementedError("disc_loss has not been implemented")

    def gen_loss(
        self,
        fake_preds: torch.Tensor,
        real_preds: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """[summary]

        Parameters
        ----------
        fake_preds : torch.Tensor
            output of the discriminator for fake samples, shape = [batch_size, ]
        real_preds : torch.Tensor, optional
            output of the discriminator for real samples, shape = [batch_size], by default None

        Returns
        -------
        torch.Tensor
            return the generator loss

        Raises
        ------
        NotImplementedError
            child class should implement this method
        """
        raise NotImplementedError("gen_los loss has not been implemented")


class StandardGANLoss(Loss):
    """
    Implements the standard generator GAN loss proposed in:
    https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf
    """

    def __init__(self, name="standard", **kwargs) -> None:
        super().__init__(name, **kwargs)

    def disc_loss(
        self, real_preds: torch.Tensor, fake_preds: torch.Tensor
    ) -> torch.Tensor:
        """Compute the discriminator forward step loss

        Parameters
        ----------
        real_preds : torch.Tensor
            raw discriminator predictions for real samples, shape=[batch_size,]
        fake_preds : torch.Tensor
            raw discriminator preditioncs for fake samples, shape=[batch_size,]

        Returns
        -------
        torch.Tensor
            return the computed loss
        """
        # use softplus function softplus(x) = log(1 + exp(x))
        # for more details see: https://pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html#torch.nn.functional.softplus
        # This is equivalent to nn.BCEWithLogitsLoss() in more shorter form
        real_loss = F.softplus(-real_preds).mean()
        fake_loss = F.softplus(fake_preds).mean()

        return (real_loss + fake_loss) / 2

    def gen_loss(self, fake_preds: torch.Tensor) -> torch.Tensor:
        """Compute the loss when optimizing the generator

        Parameters
        ----------
        fake_preds : torch.Tensor
            raw discriminator predictions for fake samples, shape=[batch_size,]

        Returns
        -------
        torch.Tensor
            return the computed loss
        """
        return -1 * F.softplus(fake_preds).mean()


class NSGANLoss(StandardGANLoss):
    """
    Implements non-saturating version of the standard loss proposed in:
    https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf
    It has the same implementation for the discriminator, only the generator loss changes
    """

    def __init__(self, name="non_saturating", **kwargs) -> None:
        super().__init__(name, **kwargs)

    def gen_loss(self, fake_preds: torch.Tensor) -> torch.Tensor:
        """Compute the loss when optimizing the generator

        Parameters
        ----------
        fake_preds : torch.Tensor
            raw discriminator predictions for fake samples, shape=[batch_size,]

        Returns
        -------
        torch.Tensor
            return the computed loss
        """
        return F.softplus(-fake_preds).mean()


class WassersteinGANLoss(Loss):
    """
    Implements the Wasserstein loss proposed in:
    http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf
    """

    def __init__(self, name="wassertein", **kwargs) -> None:
        super().__init__(name, **kwargs)

    def disc_loss(
        self, fake_preds: torch.Tensor, real_preds: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute the loss when optimizing the discriminator

        Parameters
        ----------
        real_preds : torch.Tensor
            raw discriminator predictions for real samples, shape=[batch_size,]
        fake_preds : torch.Tensor
            raw discriminator preditioncs for fake samples, shape=[batch_size,]

        Returns
        -------
        torch.Tensor
            return the computed loss
        """
        return -real_preds.mean() + fake_preds.mean()

    def gen_loss(self, fake_preds: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the loss when optimizing the generator

        Parameters
        ----------
        real_preds : torch.Tensor
            raw discriminator predictions for real samples, shape=[batch_size,]

        Returns
        -------
        torch.Tensor
            return the computed loss
        """
        return -fake_preds.mean()


class WassersteinLossWithGP(WassersteinGANLoss):
    """
    Implements Wasserstein loss with gradient penalty proposed in:
    https://proceedings.neurips.cc/paper/2017/file/892c3b1c6dccd52936e27cbd0ff683d6-Paper.pdf
    """

    def __init__(self, gp_weight, name="wassertein_gp", **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.gp_weight = gp_weight

    def disc_loss(
        self,
        fake_preds: torch.Tensor,
        real_preds: torch.Tensor,
        fake_samples: torch.Tensor,
        real_samples: torch.Tensor,
        discriminator: nn.Module,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the discriminator loss

        Parameters
        ----------
        fake_preds : torch.Tensor
            the dicriminator's scores of the fake images
        real_preds : torch.Tensor
            the dicriminator's scores of the real images
        fake_samples : torch.Tensor
            batch of fake samples of shape = [batch_size, **data_dim]
        real_samples : torch.Tensor
            batch of real samples of shape = [batch_size, **data_dim]
        discriminator : nn.Module
            discriminator/critor network that takes real_samples/fake_samples and returs a float value for each sample

        Returns
        -------
        torch.Tensor
            return the computed the loss
        """
        epsilon = torch.rand(
            len(real_samples), 1, 1, 1, device=real_samples.device, requires_grad=True
        )
        # Mix the images together
        mixed_images = real_samples * epsilon + fake_samples * (1 - epsilon)

        # Calculate the critic's scores on the mixed images
        mixed_scores = discriminator(mixed_images)

        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=mixed_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(len(gradient), -1)
        gradient_norm = gradient.norm(2, dim=1)
        penalty = torch.mean((gradient_norm - 1) ** 2)
        return torch.mean(fake_preds) - torch.mean(real_preds) + self.gp_weight * penalty


class LSGANLoss(Loss):
    """
    Implements Least Sqaures GAN loss proposed in:
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Mao_Least_Squares_Generative_ICCV_2017_paper.pdf
    """

    def __init__(self, name="least_squares", **kwargs) -> None:
        super().__init__(name, **kwargs)

    def disc_loss(
        self, fake_preds: torch.Tensor, real_preds: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute the discriminator loss

        Parameters
        ----------
        fake_preds : torch.Tensor
            the dicriminator's scores of the fake images
        real_preds : torch.Tensor
            the dicriminator's scores of the real images

        Returns
        -------
        torch.Tensor
            return the computed loss
        """
        return 0.5 * ((-real_preds - 1.0).pow(2).mean() + fake_preds.pow(2).mean())

    def gen_loss(self, fake_preds: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the generator loss

        Parameters
        ----------
        fake_preds : torch.Tensor
            the dicriminator's scores of the fake images

        Returns
        -------
        torch.Tensor
            return the computed loss
        """

        return -0.5 * (fake_preds - 1.0).pow(2).mean()


class HingeLoss(Loss):
    """
    Implements the Hinge GAN loss proposed in:
    https://arxiv.org/pdf/1705.02894.pdf
    """

    def __init__(self, name="hinge_loss"):
        super().__init__(name)

    def disc_loss(
        self, fake_preds: torch.Tensor, real_preds: torch.Tensor
    ) -> torch.Tensor:
        """Compute the discriminator loss

        Parameters
        ----------
        fake_preds : torch.Tensor
            the dicriminator's scores of the fake images
        real_preds : torch.Tensor
            the dicriminator's scores of the real images

        Returns
        -------
        torch.Tensor
            loss value
        """
        return torch.mean(nn.ReLU()(1 - real_preds)) + torch.mean(
            nn.ReLU()(1 + fake_preds)
        )

    def gen_loss(self, fake_preds: torch.Tensor) -> torch.Tensor:
        """Compute the generator loss

        Parameters
        ----------
        fake_preds : torch.Tensor
            the dicriminator's scores of the fake images

        Returns
        -------
        torch.Tensor
            loss value
        """
        return -torch.mean(fake_preds)


class RelativisticAverageHingeLoss(Loss):
    """
    Implement the Relativiste Average Hinge GAN loss proposed in:

    """

    def __init__(self, name="relativistic_hinge"):
        super().__init__()
        self.name = name

    def disc_loss(
        self, fake_preds: torch.Tensor, real_preds: torch.Tensor
    ) -> torch.Tensor:
        """Compute the discriminator loss

        Parameters
        ----------
        fake_preds : torch.Tensor
            the dicriminator's scores of the fake images
        real_preds : torch.Tensor
            the dicriminator's scores of the real images

        Returns
        -------
        torch.Tensor
            loss value
        """
        real_fake_difference = real_preds - torch.mean(fake_preds)
        fake_real_difference = fake_preds - torch.mean(real_preds)
        return torch.mean(nn.ReLU()(1 - real_fake_difference)) + torch.mean(
            nn.ReLU()(1 + fake_real_difference)
        )

    def gen_loss(
        self, fake_preds: torch.Tensor, real_preds: torch.Tensor
    ) -> torch.Tensor:
        """Compute the generator loss

        Parameters
        ----------
        fake_preds : torch.Tensor
            the dicriminator's scores of the fake images
        real_preds : torch.Tensor
            the dicriminator's scores of the real images

        Returns
        -------
        torch.Tensor
            loss value
        """
        real_fake_difference = real_preds - torch.mean(fake_preds)
        fake_real_difference = fake_preds - torch.mean(real_preds)

        return torch.mean(nn.ReLU()(1 + real_fake_difference)) + torch.mean(
            nn.ReLU()(1 - fake_real_difference)
        )


class DRAGANLoss(Loss):
    """
    Implements the generator loss proposed in:
    https://arxiv.org/pdf/1705.07215.pdf
    """

    def __init__(self, weight, name="dra_loss", **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.weight = weight

    def disc_loss(
        self,
        fake_preds: torch.Tensor,
        real_preds: torch.Tensor,
        real_samples: torch.Tensor,
        discriminator: nn.Module,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the discriminator loss

        Parameters
        ----------
        fake_preds : torch.Tensor
            the dicriminator's scores of the fake images
        real_preds : torch.Tensor
            the dicriminator's scores of the real images
        real_samples : torch.Tensor
            batch of real samples of shape = (batch_size, d1, d2, )
        discriminator : nn.Module
            discriminator/critor network that takes real_samples/fake_samples and returs a float value for each sample

        Returns
        -------
        torch.Tensor
            return the computed the loss
        """
        loss = F.softplus(-real_preds).mean() + F.softplus(fake_preds).mean()

        # Compute regularization loss
        real_prediction = discriminator(
            real_samples + torch.randn_like(real_samples) * 0.1
        )
        grad_real = torch.autograd.grad(
            outputs=real_prediction.sum(), inputs=real_samples, create_graph=True
        )[0]
        loss = loss + self.weight * (grad_real.norm(dim=1) - 1.0).pow(2)

    def gen_loss(self, fake_preds: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the generator loss

        Parameters
        ----------
        fake_preds : torch.Tensor
            the dicriminator's scores of the fake images

        Returns
        -------
        torch.Tensor
            loss value
        """
        return -F.softplus(fake_preds).mean()


class R1Regularizer(nn.Module):
    """Implements R1 GAN Regularizer"""

    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(
        self, real_preds: torch.Tensor, real_samples: torch.Tensor
    ) -> torch.Tensor:
        """Compute the R1 regularization

        Parameters
        ----------
        real_preds : torch.Tensor
            the dicriminator's scores of the real images of shape = (batch_size, )
        real_samples : torch.Tensor
            a batch of real samples of shape = (batch_size, d1, d2, )

        Returns
        -------
        torch.Tensor
            regularization value
        """
        gradient = torch.autograd.grad(
            outputs=real_preds.sum(), inputs=real_samples, create_graph=True
        )[0]

        return self.weight * gradient.pow(2).view(gradient.shape[0], -1).sum(1).mean()


class R2Regularizer(nn.Module):
    """Implements R2 GAN Regularizer"""

    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(
        self, fake_preds: torch.Tensor, fake_samples: torch.Tensor
    ) -> torch.Tensor:
        """Compute the R1 regularization

        Parameters
        ----------
        fake_preds : torch.Tensor
            the dicriminator's scores of the real images
        fake_samples : torch.Tensor
            a batch of real samples

        Returns
        -------
        torch.Tensor
            regularization value
        """
        gradient = torch.autograd.grad(
            outputs=fake_preds.sum(), inputs=fake_samples, create_graph=True
        )[0]

        return self.weight * gradient.pow(2).view(gradient.shape[0], -1).sum(1).mean()


class RLCRegularizer(nn.Module):
    """
    Implements of the RLC GAN regularization proposed in:
    https://arxiv.org/pdf/2104.03310.pdf
    """

    def __init__(self, rlc_af, rlc_ar, rlc_w):
        super().__init__()
        self.rlc_af = rlc_af
        self.rlc_ar = rlc_ar
        self.rlc_w = rlc_w

    def forward(
        self,
        real_preds: torch.Tensor,
        fake_preds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the RLC regularization

        Parameters
        ----------
        fake_preds : torch.Tensor
            the dicriminator's scores of the real images
        fake_samples : torch.Tensor
            a batch of real samples

        Returns
        -------
        torch.Tensor
            regularization value
        """

        regularization_loss = (real_preds - self.rlc_af).norm(dim=-1).pow(2).mean() + (
            fake_preds - self.rlc_ar
        ).norm(dim=-1).pow(2).mean()
        return self.rlc_w * regularization_loss


class LogisticLoss(Loss):
    def __init__(self, name="logistic", weight=5, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.loss = NSGANLoss()
        self.regularizer = R1Regularizer(weight=weight)

    def gen_loss(self, fake_preds: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.loss.gen_loss(fake_preds)

    def disc_loss(
        self,
        fake_preds: torch.Tensor,
        real_preds: torch.Tensor,
        real_samples: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        loss = self.loss.disc_loss(real_preds, fake_preds) + self.regularizer(
            real_preds, real_samples
        )
        return loss
