
# Skin lesions classification using machine learning techniques. 

In this project we were asked to classify images from a skin lesion dataset with the constraint of using only ML techniques. On the one hand, we had to classify nevus images vs. other type of images: to do that we had an approximately balanced train set of about 15000 images and a validation set of about 4000 images. On the other hand, we were asked to classify images on Melanoma, Basal and Squamous Cell Cancers. However, in this case the datasets are unbalanced, for a training set of circa 5000 images and a validation set of approx 1000 images on a proportion of 50:40:10. We were provided with respective test sets to see the performance of our implementation.

Following, we will show you all the trials we did before finding the final pipeline. We focused on the first weeks on the pre-processing. The first step consisted of cropping the images if they had a microscope halo by detecting dark corners. However, if the cropping reduces too much the image, a more flexible cropping is applied. Then, we proceeded to resize the image, extract the luminance channel, apply the Contrast Limited Adaptive Histogram Equalization for enhancement, noise reduction, and we tried to mask and remove the hairs in the skin. Finally, we normalized the images and applied an Expectation-Maximization algorithm to cluster lesion and non-lesion tissues, and applied a hole filling algorithm. Sometimes, the mask had too few pixels, in which case we simply cropped the original image; in other cases, the mask had too many pixels because it turned the non-lesion tissue as a lesion, so we simply inverted it.

![presentation_page-0004](https://github.com/amina-bzd/melanoma_classification/assets/57720297/07f19198-ba4f-4d0d-8868-a9f3cc03c42e)

Here we want to display some examples on these pre-processing steps, and show that sometimes worked and sometimes not. For example, the cropping generally removes the halo correctly, but sometimes leaves some remnants. It happens too with the hair removal: sometimes there is not much change. Finally, we can also see that the masking sometimes worked, although giving some weird shapes, but in many cases the result was not acceptable, for example due to non-removed hairing.

![presentation_page-0005](https://github.com/amina-bzd/melanoma_classification/assets/57720297/620fd6a8-7716-49d0-ac3e-5bfd05452a3a)

### Final Pipeline: Binary Classification

![presentation_page-0010](https://github.com/amina-bzd/melanoma_classification/assets/57720297/eb4c9f92-552b-4b48-9ee3-140a4371c14f)

### Final Pipeline: Three-Class Classification

![presentation_page-0011](https://github.com/amina-bzd/melanoma_classification/assets/57720297/2267cdcf-db6d-4fc9-b82a-350ba66e4605)


### Conclusions & Limitations

![presentation_page-0012](https://github.com/amina-bzd/melanoma_classification/assets/57720297/dd85abe1-ecaa-427e-854f-df875896b675)










