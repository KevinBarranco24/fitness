"""
Main script.
Create panoramas from a set of images.
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from src.images import Image
from src.matching import (
    MultiImageMatches,
    PairMatch,
    build_homographies,
    find_connected_components,
)
from src.rendering import multi_band_blending, set_gain_compensations, simple_blending

parser = argparse.ArgumentParser(
    description="Create panoramas from a set of images. \
                 All the images must be in the same directory. \
                 Multiple panoramas can be created at once"
)

parser.add_argument(dest="data_dir", type=Path, help="directory containing the images")
parser.add_argument(
    "-mbb",
    "--multi-band-blending",
    action="store_true",
    help="use multi-band blending instead of simple blending",
)
parser.add_argument(
    "--size", type=int, help="maximum dimension to resize the images to"
)
parser.add_argument(
    "--num-bands", type=int, default=5, help="number of bands for multi-band blending"
)
parser.add_argument(
    "--mbb-sigma", type=float, default=1, help="sigma for multi-band blending"
)

parser.add_argument(
    "--gain-sigma-n", type=float, default=10, help="sigma_n for gain compensation"
)
parser.add_argument(
    "--gain-sigma-g", type=float, default=0.1, help="sigma_g for gain compensation"
)

parser.add_argument(
    "-v", "--verbose", action="store_true", help="increase output verbosity"
)

args = vars(parser.parse_args())

if args["verbose"]:
    logging.basicConfig(level=logging.INFO)

logging.info("Gathering images...")

valid_images_extensions = {".jpg", ".png", ".bmp", ".jpeg"}

image_paths = [
    str(filepath)
    for filepath in args["data_dir"].iterdir()
    if filepath.suffix.lower() in valid_images_extensions
]
#AQUI CARGA LAS IMAGENES
images = [Image(path, args.get("size")) for path in image_paths]

logging.info("Found %d images", len(images))
logging.info("Computing features with SIFT...")

for image in images:
    
    image.compute_features()

logging.info("Matching images with features...")

matcher = MultiImageMatches(images)
pair_matches: list[PairMatch] = matcher.get_pair_matches()
pair_matches.sort(key=lambda pair_match: len(pair_match.matches), reverse=True)

logging.info("Finding connected components...")

connected_components = find_connected_components(pair_matches)

logging.info("Found %d connected components", len(connected_components))
logging.info("Building homographies...")

build_homographies(connected_components, pair_matches)

time.sleep(0.1)

logging.info("Computing gain compensations...")

for connected_component in connected_components:
    component_matches = [
        pair_match
        for pair_match in pair_matches
        if pair_match.image_a in connected_component
    ]

    set_gain_compensations(
        connected_component,
        component_matches,
        sigma_n=args["gain_sigma_n"],
        sigma_g=args["gain_sigma_g"],
    )

    

time.sleep(0.1)

for image in images:
    image.image = (image.image * image.gain[np.newaxis, np.newaxis, :]).astype(np.uint8)

results = []

if args["multi_band_blending"]:
    logging.info("Applying multi-band blending...")
    results = [
        multi_band_blending(
            connected_component,
            num_bands=args["num_bands"],
            sigma=args["mbb_sigma"],
        )
        for connected_component in connected_components
    ]


else:
    logging.info("Applying simple blending...")
    results = [
        simple_blending(connected_component)
        for connected_component in connected_components
    ]

logging.info("Saving results to %s", args["data_dir"] / "results")

I_R = []
I_G = []
I_B = []

(args["data_dir"] / "results").mkdir(exist_ok=True, parents=True)

for i, result in enumerate(results):
    """
    I_R = np.zeros(result.shape)
    I_G = np.zeros(result.shape)
    I_B = np.zeros(result.shape)
    for x in range(0,result.shape[0]):
        for y in range(0,result.shape[1]):
            I_B[x][y][0] = result[x][y][0]
            I_G[x][y][1] = result[x][y][1]
            I_R[x][y][2] = result[x][y][2]
    
    cv2.imshow("Rojo", I_R)
    cv2.imshow("Rojo", I_G)
    cv2.imshow("Rojo", I_B)
    
    I_R = result.copy()
    I_G = result.copy()
    I_B = result.copy()

    I_B[:,:,1]=0
    I_B[:,:,2]=0

    I_G[:,:,0]=0
    I_G[:,:,2]=0

    I_R[:,:,0]=0
    I_R[:,:,1]=0

    #M_R
    #Obtener filas y columnas
    R = np.array(I_R.shape[0])
    L = np.array(I_R.shape[1])
    intenR = np.zeros(1)
    CR = 1/(R*L)

    for i in range(1,R):
        for j in range(1,L):
            intenR = intenR + I_R[i][j][2]
        
    
    M_R = CR*intenR;

    #M_G
    #Obtener filas y columnas
    R = np.array(I_G.shape[0])
    L = np.array(I_G.shape[1])
    intenG = np.zeros(1)
    CG = 1/(R*L)

    for i in range(1,R):
        for j in range(1,L):
            intenG = intenG + I_G[i][j][1]
        
    M_G = CG*intenG;

    #M_B
    #Obtener filas y columnas
    R = np.array(I_B.shape[0])
    L = np.array(I_B.shape[1])
    intenB = np.zeros(1)
    CB = 1/(R*L)

    for i in range(1,R):
        for j in range(1,L):
            intenB = intenB + I_B[i][j][0]

    M_B = CB*intenB;

    #Mean Value
    Fit = (M_R + M_G + M_B)/3;

    print("M_R ", M_R)
    print("M_G ", M_G)
    print("M_B ", M_B)
    print("Mean Value ", Fit)
    
    #cv2.imwrite(str(args["data_dir"] / "rojo.jpg"), I_R)
    #cv2.imwrite(str(args["data_dir"] / "verde.jpg"), I_G)
    #cv2.imwrite(str(args["data_dir"] / "azul.jpg"), I_B)
    """
    cv2.imwrite(str(args["data_dir"] / "results" / f"pano_{i}.jpg"), result)
