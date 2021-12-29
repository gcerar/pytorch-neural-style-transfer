# Neural Style Transfer

The repository contains PyTorch implementation of the [Gatys *et al.*](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) paper with some modifications. The authors propose algorithm that transfers style from one input image (the *style* image) into second image (the *content* image).

## Examples

Some of the hand picked results:

![The green bridge + Walk in the rain](./examples/bridge-rainwalk.jpg)
![The green bridge + Starry night](./examples/bridge-starry.jpg)
![The green bridge + Bronze leaf pattern](./examples/bridge-bronze.jpg)
![The green bridge + Rainbow swirl](./examples/bridge-swirl.jpg)
![Breakfast + Mosaic](./examples/breakfast-mosaic.jpg)
![Breakfast + Bronze leaf lattern](./examples/breakfast-bronze.jpg)
![Nature + Candy](./examples/nature-candy.jpg)
![Nature + Starry night](./examples/nature-starry.jpg)

## Setup

1. Clone the repository `git clone https://github.com/gcerar/pytorch-neural-style-transfer`
2. Enter directory `cd pytorch-neural-style-transfer`
3. Run `conda env create -f ./environment.yml`
4. Run `conda activate style-transfer`

It is recommended to use GPU to accelerate the style transfer process.

## Usage

`python style_transfer.py --input <TargetImgPath> --style <StyleImgPath>`

Optional parameters:

- `--help` or `-h` print help with all available parameters
- `--epochs <number>` (default: 7000)
- `--seed <number>` (default: 1)
- `--no-cuda` disables CUDA acceleration and use CPU instead
- `--optimizer <adam|adamw|lbfgs|sgd>` (default: adam)
- `--init <input|noise>` decide on what is init of output image (default: input)

## Acknowledgment

Useful articles with code and code repositories:

- Gregor Koehler *et al.* [gkoehler/pytorch-neural-style-transfer](https://nextjournal.com/gkoehler/pytorch-neural-style-transfer) (best resource in my opinion)
- Ritul's [Medium article](https://medium.com/udacity-pytorch-challengers/style-transfer-using-deep-nural-network-and-pytorch-3fae1c2dd73e) (good resource)
- Pragati Baheti [blog](https://www.v7labs.com/blog/neural-style-transfer) visually present style extraction
- Aleksa Gordić ([gordicaleksa/pytorch-neural-style-transfer](https://github.com/gordicaleksa/pytorch-neural-style-transfer))
- [ProGamerGov/neural-style-pt](https://github.com/ProGamerGov/neural-style-pt/blob/master/neural_style.py)
- Katherine Crowson ([rowsonkb/style-transfer-pytorch](https://github.com/crowsonkb/style-transfer-pytorch/blob/master/style_transfer/style_transfer.py))
- Derrick Mwiti's [Medium article](https://heartbeat.comet.ml/neural-style-transfer-with-pytorch-49e7c1fe3bea)
- Aman Kumar Mallik's [Medium article](https://towardsdatascience.com/implementing-neural-style-transfer-using-pytorch-fd8d43fb7bfa)

Sample *content* images:

<div style="display:flex;flex-flow:row wrap;align-items:flex-end;text-align:center;">

<figure>
    <img src="./samples/bridge.jpg" style="max-height:270px; max-width:270px;">
    <figcaption>"Gray Bridge and Trees", Martin Damboldt</figcaption>
</figure>

</div>


Style images:

<div style="display:flex;flex-flow:row wrap;align-items:flex-end;text-align:center;">
<figure>
    <img src="./styles/bamboo-forest.jpg" style="max-height:270px; max-width:270px;">
    <figcaption>Bamboo forest, ???</figcaption>
</figure>

<figure>
    <img src="./styles/candy.jpg" style="max-height:270px; max-width:270px;">
    <figcaption>"June tree", Natasha (Wescoat) Bouchillion</figcaption>
</figure>

<figure>
    <img src="./styles/colorful-whirlpool.jpg" style="max-height:270px; max-width:270px;">
    <figcaption>Colorful whirlpool, ???</figcaption>
</figure>

<figure>
    <img src="./styles/composition-vii.jpg" style="max-height:270px; max-width:270px;">
    <figcaption>"Composition VII", Wassily Kandinsky</figcaption>
</figure>

<figure>
    <img src="./styles/doomguy.jpg" style="max-height:270px; max-width:270px;">
    <figcaption>"Doomguy", ID Software</figcaption>
</figure>

<figure>
    <img src="./styles/edtaonisl.jpg" style="max-height:270px; max-width:270px;">
    <figcaption>"Edtaonisl", Francis Picabia</figcaption>
</figure>

<figure>
    <img src="./styles/escher-sphere.jpg" style="max-height:270px; max-width:270px;">
    <figcaption>"Hand with Reflecting Sphere", M. C. Escher</figcaption>
</figure>

<figure>
    <img src="./styles/feathers.jpg" style="max-height:270px; max-width:270px;">
    <figcaption>Feather leaves and petals, Kathryn Corlett</figcaption>
</figure>

<figure>
    <img src="./styles/flowers.jpg" style="max-height:270px; max-width:270px;">
    <figcaption>Flowers, ???</figcaption>
</figure>

<figure>
    <img src="./styles/fractal-pattern.jpg" style="max-height:270px; max-width:270px;">
    <figcaption>Fractal Patterns, ???</figcaption>
</figure>

<figure>
    <img src="./styles/la-muse.jpg" style="max-height:270px; max-width:270px;">
    <figcaption>"La Muse", Pablo Picasso</figcaption>
</figure>

<figure>
    <img src="./styles/lady.jpg" style="max-height:270px; max-width:270px;">
    <figcaption>"Lady", ???</figcaption>
</figure>

<figure>
    <img src="./styles/mondrian.png" style="max-height:270px; max-width:270px;">
    <figcaption>"Mondrian World Map", Michael Tompsett</figcaption>
</figure>

<figure>
    <img src="./styles/night-cafe.jpg" style="max-height:270px; max-width:270px;">
    <figcaption>"The Night Café", Vincent van Gogh</figcaption>
</figure>

<figure>
    <img src="./styles/paul-of-tarsus.jpg" style="max-height:270px; max-width:270px;">
    <figcaption>"Saul of Tarsus", Marko Ivan Rupnik</figcaption>
</figure>

<figure>
    <img src="./styles/persistance-of-memory.jpg" style="max-height:270px; max-width:270px;">
    <figcaption>"The Persistence of Memory", Salvador Dali</figcaption>
</figure>


<figure>
    <img src="./styles/rain-princess.jpg" style="max-height:270px; max-width:270px;">
    <figcaption>"Mysterious Rain Princess", Leonid Afremov</figcaption>
</figure>

<figure>
    <img src="./styles/seated-nude.jpg" style="max-height:270px; max-width:270px;">
    <figcaption>"Seated Nude", Pablo Picasso</figcaption>
</figure>

<figure>
    <img src="./styles/starry-night.jpg" style="max-height:270px; max-width:270px;">
    <figcaption>"The Starry Night", Vincent van Gogh</figcaption>
</figure>

<figure>
    <img src="./styles/the-scream.jpg" style="max-height:270px; max-width:270px;">
    <figcaption>"The Scream", Edvard Munch</figcaption>
</figure>

<figure>
    <img src="./styles/udnie.jpg" style="max-height:270px; max-width:270px;">
    <figcaption>"Udnie", Francis Picabia</figcaption>
</figure>

<figure>
    <img src="./styles/walking-in-the-rain.jpg" style="max-height:270px; max-width:270px;">
    <figcaption>"Walking in the Rain", Leonid Afremov</figcaption>
</figure>

<figure>
    <img src="./styles/wave.jpg" style="max-height:270px; max-width:270px;">
    <figcaption>"Under the Wave off Kanagawa", Katsushika Hokusai</figcaption>
</figure>

</div>