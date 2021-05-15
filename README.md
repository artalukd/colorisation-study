# colorisation-study



# Deploying Colorization on Flask server

- Ensure all dependencies in requirements.txt are met.
- Pretrained model is available in the repo base directory, the model given in this repo is UNet (based on Resnet 34) trained on landscapes dataset.
- Python version is > 3.6
- run ```python app.py```
- open http://127.0.0.1:5000/


# Model Architecture
We build a Resnet34 based U-Net architecture in pytorch. We use Upsample+Convolution layer in the upscaling blocks because Transpose convolution layers present themselves with checkerboard artifacts. The Retnet34 backbone is not truncated as done in prior work and we show that going deeper did in fact help reduce loss at a higher rate. Skip connections we implemented were by using concatenation function. There are works which use element wise summation as well. Compute cost is less in element wise summation, but flexibility for representation learning is more in concatenation as the network gets to optimise over a larger representation space.


# REFERENCES
[1] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015. [1505.04597] U-Net: Convolutional Networks for Biomedical Image Segmentation


[2] Kodali, Naveen, et al. "On convergence and stability of gans." arXiv preprint arXiv:1705.07215 (2017).[1705.07215] On Convergence and Stability of GANs


[3] Gu, Shuhang, Radu Timofte, and Richard Zhang. "Ntire 2019 challenge on image colorization: Report." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. 2019. https://ieeexplore.ieee.org/document/9025578 


[4] Cheng, Zezhou, Qingxiong Yang, and Bin Sheng. "Deep colorization." Proceedings of the IEEE International Conference on Computer Vision. 2015. [1605.00075] Deep Colorization


[5] Zhang, Richard, et al. "Real-time user-guided image colorization with learned deep priors." arXiv preprint arXiv:1705.02999 (2017).[1705.02999] Real-Time User-Guided Image Colorization with Learned Deep Priors


[6] U-Net deep learning colourisation of greyscale images


[7] Colorizing black & white images with U-Net and conditional GAN — A Tutorial


[8] Image Colorization with Convolutional Neural Networks


[9] Deploying PyTorch in Python via a REST API with Flask — PyTorch Tutorials 1.8.1+cu102 documentation


[10]Timofte, Radu, et al. "Ntire 2017 challenge on single image super-resolution: Methods and results." Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2017.


[11]Landscape Pictures https://www.kaggle.com/arnaud58/landscape-pictures 


[12]Hore, Alain, and Djemel Ziou. "Image quality metrics: PSNR vs. SSIM." 2010 20th international conference on pattern recognition. IEEE, 2010. 


