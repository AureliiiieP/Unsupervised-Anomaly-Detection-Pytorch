# Unsupervised-Anomaly-Detection-Pytorch
Use unsupervised learning for anomaly detection.

In an industrial setting (factories etc), it is usually easy to collect a lot of samples without defect. However anomalies can be very diverse in appearance, type and very rare which make it very difficult to collect and annonate enough data. Unsupervised learning is very useful in such cases.

Idea is that the model task is to reconstruct the image given as input. Since it has only been trained on "normal" samples, reconstruction on anomalies parts should be poor. By taking difference of the reconstruction output and input, we may be able to detect the anomaly.

## Toy dataset
As example, we created a small toy dataset of xx images.

Anomaly for testing include
- Missing chocolate
- Chocolat smear
- Broken cookie

- [ ] Add Unet
- [ ] Add VAE
- [ ] Add Normalizing flow (Less able to generate new content)
- [ ] Improve reproducibility (copyfile config, save datasplit)

## Ressources used 
- https://github.com/usuyama/pytorch-unet UNET architecture
