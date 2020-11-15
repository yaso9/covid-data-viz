import os
import subprocess
import shutil
import json
import multiprocessing

import pandas as pd
import numpy as np
import geopandas as gpd
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


def make_frame(frame_counter, df, gdf, day, next_day):
    font = ImageFont.truetype("Roboto-Regular.ttf", 150)
    image_path = os.path.join("images", f"{frame_counter:04d}.png")

    # Plot the map and legend
    daydf = df[df.submission_date == day].set_index("state")
    # Interpolate the values for inbetween frames
    next_daydf = df[df.submission_date == next_day].set_index("state")
    daydf.tot_cases = ((next_daydf.tot_cases - daydf.tot_cases) / 12) * (
        frame_counter % 12
    ) + daydf.tot_cases
    daygdf = gdf.join(daydf, on="STATE", how="right").set_index("STATE", drop=False)

    ax = daygdf.plot(column="tot_cases", legend=True)
    ax.set_xlim(-130, -60)
    ax.set_ylim(24, 50)
    frame = plt.gca()
    frame.axis("off")
    plt.tight_layout(pad=0, h_pad=0, w_pad=0, rect=(0, 0, 1, 1))
    plt.savefig(image_path, dpi=420)
    plt.close("all")

    # Process with pillow
    img = Image.new("RGB", (3840, 2160), (255, 255, 255))
    fg = Image.open(image_path)
    img.paste(fg, ((img.size[0] - fg.size[0]) // 2, (img.size[1] - fg.size[1]) // 2))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), np.datetime_as_string(day, unit="D"), font=font, fill=(0, 0, 0))
    img.save(image_path)
    return image_path


if __name__ == "__main__":
    # Load covid data
    df = pd.read_csv(
        "covid-data.csv",
        dtype={"submission_date": str},
        parse_dates=["submission_date", "created_at"],
    )

    # Convert the state abbreviations to fips codes
    with open("stateCodeToFips.json", "r") as file:
        state_code_to_fips = json.load(file)
    df.state = df.state.replace(state_code_to_fips)
    df.state = pd.to_numeric(df.state, errors="coerce")
    df = df[df.state.notnull()]
    df.state = df.state.astype(np.int)
    df.tot_cases = df.tot_cases.astype(np.float64)

    # Load the GeoJSON data
    gdf = gpd.read_file("state-geojson.json")
    gdf.STATE = gdf.STATE.astype(np.int)

    # Make each frame of the video
    frame_counter = 0
    pool = multiprocessing.Pool(16)
    results = []
    days = np.sort(pd.unique(df.submission_date))
    for day, next_day in zip(days, days[1:]):
        for i in range(12):
            results.append(
                pool.apply_async(make_frame, (frame_counter, df, gdf, day, next_day))
            )
            frame_counter += 1
    for result in tqdm(results):
        image_path = result.get()

    # Hold the final frame for 5 seconds
    for i in range(24 * 5):
        shutil.copyfile(image_path, os.path.join("images", f"{frame_counter}.png"))
        frame_counter += 1

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            "images/%04d.png",
            "-i",
            "music.mp3",
            "-r",
            "24",
            "-c:v",
            "hevc_nvenc",
            "-shortest",
            "out.mkv",
        ]
    )
