# ROS-Neuro gaussian decoder package

This package implements a gaussian classifier as a plugin for rosneuro::decoder::Decoder and as class. The test used it as class, for the usage as rosnode it misses the tensor message after the pwelch computation, which will be done as soon as possible.

## Usage
The package required as ros parameter:
Markup : 
    - **cfg_name**: which is the name of the structure in the yaml file
    - **yaml file**: which contains the structure for the lda classifier

## Example of yaml file
Markup: 
`
GaussianCfg:
  name: "gaussian"
  params: 
    filename: "file1"
    subject: "S1"
    nclasses: 2
    classlbs: [771, 773]
    nprototypes: 1
    nfeatures: 6
    idchans: [1, 2]
    freqs: "10 12 14; 
            10 12 14"
    covs:    "0.4529;
              0.3777;
              0.5301;
              0.4910;
              0.7136;
              0.7507;
              1.0767;
              1.1501;
              0.7055;
              0.6599;
              1.2610;
              1.1729;"
    centers: "-2.9844;
              -3.2610;
              -2.3018;
              -2.5743;
              -2.3550;
              -2.5197;
              -2.1708;
              -2.3625;
              -1.8599;
              -2.2393;
              -1.5272;
              -1.8280;"
`

Some parameters are hard coded:
Markup: 
    - **idchans**: the index of the channels from 1 to the number of channels used;
    - **freqs:**: the selected frequencies;
    - **centers**: matrix [(features * classes) x prototypes];
    - **covs**: matrix [(features * classes) x prototypes]. 