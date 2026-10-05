# YOSHINON RHYME PV

A 1:54 fan-made opening movie starring 依田芳乃. It follows the cut structure and timing of the
reference opening movie, with every character shot replaced by Yoshino artwork from this repository.

- Character art: `card*.png`, `Yoshino SSR*.png`, `tachie*.png`, `ysn*.png`
- Logos and ribbon: `logo1.jpg` (brand), `logo2.jpg` (title, background removed in code), `heart1-3.png`
- Backgrounds: `haikei.png`, `bg_sakura_night.jpg`, `bg_waterfall.jpg`
- Drawn in code: halftone screens, rotating squares, sakura petals, line-art (sketch) filter, zoom blur,
  lens glare, emotes (♪, !, sweat drop), the 芳 seal, the group/action collage backdrops and all text

Catch copy: 神さびて　愛らしく――それはひとりの偶像（神様）の物語。

## Render

```sh
pip install pillow numpy opencv-python-headless
# fonts: M PLUS 2, Noto Serif CJK JP (apt: fonts-mplus fonts-noto-cjk)
cd pv
python3 render.py                       # -> out/yoshinon_rhyme_pv.mp4 (1024x768, 30fps, silent)
python3 render.py --sheet 12.9 28.6 99.9 # stills contact sheet -> out/sheet.png
python3 render.py --from 26 --to 45     # one section
```

The video is silent. Its cuts are timed to the reference movie, so you can add your own copy of
the music with ffmpeg:

```sh
ffmpeg -i out/yoshinon_rhyme_pv.mp4 -i <your_audio_source> -map 0:v -map 1:a -c:v copy -c:a aac -shortest out/with_audio.mp4
```

`build/` is a cache (downscaled assets, cutouts, sketches) and can be deleted at any time.

| Time | Scene |
| --- | --- |
| 0:00 | Fade in, pink line-art close-up, CINDERELLA GIRLS logo |
| 0:07 | STARRING card |
| 0:10 | Night sakura avenue, quick close-ups, catch copy |
| 0:14 | All-Yoshino group shot, ribbon heart, title logo |
| 0:26 | Five character intros (SSR1–5) with name plates |
| 0:44 | Card montage |
| 1:20 | Waterfall |
| 1:25 | Credits |
| 1:39 | Glowing ribbon and title logo |
| 1:48 | Brand logo, fade out |
