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

| ![Gray Bridge and Trees](./samples/bridge.jpg) |
|:--:|
| "Gray Bridge and Trees", Martin Damboldt |

Style images:

| ![The Persistence of Memory](./styles/persistance-of-memory.jpg) | ![Colorful whirlpool](./styles/colorful-whirlpool.jpg) | ![Mondrian World Map](./styles/mondrian.png) |
|:--:|:--:|:--:|
| "The Persistence of Memory", Salvador Dali | Colorful whirlpool, ??? | "Mondrian World Map", Michael Tompsett |

| ![Fractal Patterns](./styles/fractal-pattern.jpg) | ![The Scream](./styles/the-scream.jpg) | ![Udnie](./styles/udnie.jpg) |
|:--:|:--:|:--:|
| Fractal Patterns, ??? | "The Scream", Edvard Munch | "Udnie", Francis Picabia |

| ![Hand with Reflecting Sphere](./styles/escher-sphere.jpg) | ![Bamboo forest](./styles/bamboo-forest.jpg) | ![Mysterious Rain Princess](./styles/rain-princess.jpg) |
|:--:|:--:|:--:|
| "Hand with Reflecting Sphere", M. C. Escher | Bamboo forest, ??? | "Mysterious Rain Princess", Leonid Afremov |

| ![June tree](./styles/candy.jpg) | ![La Muse](./styles/la-muse.jpg) | ![Lady](./styles/lady.jpg) |
|:--:|:--:|:--:|
| "June tree", Natasha (Wescoat) Bouchillion | "La Muse", Pablo Picasso | "Lady", ??? |

| ![Composition VII](./styles/composition-vii.jpg) | ![Walking in the Rain](./styles/walking-in-the-rain.jpg) | ![The Starry Night](./styles/starry-night.jpg) |
|:--:|:--:|:--:|
| "Composition VII", Wassily Kandinsky | "Walking in the Rain", Leonid Afremov | "The Starry Night", Vincent van Gogh |

| ![Edtaonisl](./styles/edtaonisl.jpg) | ![Seated Nude](./styles/seated-nude.jpg) | ![The Night Café](./styles/night-cafe.jpg) |
|:--:|:--:|:--:|
| "Edtaonisl", Francis Picabia | "Seated Nude", Pablo Picasso | "The Night Café", Vincent van Gogh |

| ![Under the Wave off Kanagawa](./styles/wave.jpg) | ![Flowers](./styles/flowers.jpg) | ![Doomguy](./styles/doomguy.jpg) |
|:--:|:--:|:--:|
| "Under the Wave off Kanagawa", Katsushika Hokusai | Flowers, ??? | "Doomguy", ID Software |
