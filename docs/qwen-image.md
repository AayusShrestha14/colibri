# Qwen-Image-2.1 on colibri

`c/qwenimage` turns a text prompt into a picture with
[`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1), read
straight from the official diffusers checkpoint. There is no conversion step:
the engine quantizes the text encoder and the diffusion transformer to int8
while it loads them.

The same front ends as the text models reach it:

- `coli chat` is an image session: type a description, watch the progress,
  and the picture is drawn **inside the terminal**, then saved as a PNG.
- `coli run "prompt"` makes one image and writes it to a file.
- `coli serve` (and `coli web`) expose `POST /v1/images/generations`, the
  OpenAI images API.

## Download

The checkpoint is about 33 GB: a text encoder (Qwen3-VL, 17.5 GB bf16), the
diffusion transformer (14.2 GB bf16), the VAE (1.4 GB f32), the tokenizer and
the scheduler, each in its own folder.

```sh
hf download Qwen/Qwen-Image-2.1 --local-dir ~/Models/Qwen-Image-2.1
make -C c qwenimage
```

Keep the layout as downloaded: coli recognises the model from
`model_index.json` at the root of the folder, and there is no `config.json`
there to add. `coli info --model ~/Models/Qwen-Image-2.1` lists the parts it
found and their size on disk.

## RAM

The text encoder's language model and the transformer are held as int8 with
one scale per row; the text encoder's vision tower and its output head are
never loaded. On the real checkpoint `coli plan` reports (measured from the
safetensors headers):

| part | on disk | in RAM |
|---|---|---|
| text encoder | 17.5 GB bf16 | 7.6 GB int8 |
| diffusion transformer | 14.2 GB bf16 | 7.1 GB int8 |
| VAE | 1.4 GB f32 | 1.4 GB f32 |
| **all resident** | | **16.0 GB** |
| **text encoder on demand** | | **8.5 GB peak** |

With the text encoder on demand it is loaded for each prompt and freed before
denoising starts, so the peak is the larger of the two phases, at the cost of
reading it again for every prompt. Working buffers (the activations, the
VAE's full-resolution feature maps) come on top of both numbers:
TBD (being measured).

```sh
coli plan --model ~/Models/Qwen-Image-2.1          # both schedules against your free RAM
coli plan --model ~/Models/Qwen-Image-2.1 --json   # the same as JSON
```

## Chat: images in the terminal

```sh
coli chat --model ~/Models/Qwen-Image-2.1
```

Every line you type is a prompt. While the engine works, a status line shows
the stage (encoding the prompt, denoising step k of N, decoding) and the
elapsed time; when the engine sends previews, a small version of the picture
is redrawn in place above it. Then the final image is drawn in the terminal
and saved. Ctrl-C stops the current image (the engine finishes the step it is
on and reports it cancelled); a second Ctrl-C leaves.

Commands (TAB completes them, and the sizes after `/size`):

| command | effect |
|---|---|
| `/size WxH` | image size; `/size` alone lists the presets |
| `/steps N` | denoising steps: more is slower and usually cleaner |
| `/seed N` | fixed seed: the same prompt then gives the same image |
| `/seed random` | a new seed for every image (the default) |
| `/render MODE` | how images are drawn: `kitty`, `iterm`, `sixel`, `blocks`, `none` |
| `/help`, `/quit` | the list, and leave (`:q` works too) |

The first size, steps and seed can also be given as flags:
`coli chat --model ... --size 1024x576 --steps 12 --seed 42`.

If a `coli serve` with this model is already running on `127.0.0.1:8000`,
`coli chat` attaches to it instead of loading a second engine (the model stays
loaded after you quit). `--attach URL` picks another server, `--no-attach`
forces a private engine.

## One image from the command line

```sh
coli run --model ~/Models/Qwen-Image-2.1 --size 1024x576 --seed 42 \
  --out lighthouse.png "a lighthouse on a cliff at dusk, oil painting"
```

Without `--out` the PNG goes to the images folder. When the output is a
terminal the picture is drawn there too; when it is piped, `coli run` prints
only the path on stdout.

## Sizes, steps, seed

- **Size**: both sides a multiple of 32, from 256 to 2048. The engine groups
  image tokens 2x2, which is why 32 and not 16: 16:9 is 1024x576 or 512x288,
  never 768x432. The default is 768x512. The presets, the same ones the web UI
  offers: 512x512, 768x512, 512x768, 1024x576, 576x1024, 1024x1024.
- **Steps**: default 8, from 1 to 200.
- **Seed**: an integer from 0 to 4294967295. When you do not choose one, a
  random seed is drawn and printed with the image, so any picture can be made
  again. Seeds are the engine's own: the same seed does not give the same image
  as the Python diffusers pipeline.

## Where the images go

Every image is saved as a PNG (RGBA) in `~/colibri-images`, named after the
time, the prompt and the seed:

```
~/colibri-images/20260927-153012-a-lighthouse-on-a-cliff-at-dusk-s42.png
```

`COLI_IMAGE_DIR=/some/folder` puts them elsewhere. Nothing is overwritten: a
second image with the same name gets `-2`, `-3`, ...

## Terminal support

coli picks the best way to draw that your terminal supports, and falls back to
Unicode half blocks, which work in any terminal with colour.

| terminal | how the image is drawn |
|---|---|
| kitty, Ghostty | kitty graphics protocol (full resolution, transparency) |
| iTerm2, WezTerm | iTerm2 inline images (full resolution) |
| VS Code terminal | iTerm2 inline images when `terminal.integrated.enableImages` is on, half blocks otherwise |
| Windows Terminal 1.22 or newer (also from WSL) | sixel, when the terminal reports it; half blocks otherwise |
| foot, mlterm, xterm with sixel, konsole, mintty | sixel, when the terminal reports it |
| tmux, screen | half blocks (graphics need tmux passthrough) |
| anything else with colour | half blocks, two pixels per character cell, 24-bit colour |

Sixel support is checked by asking the terminal (a device-attributes query
with a timeout of 0.4 s); a terminal that does not answer in time gets half
blocks. Override the choice with `COLI_IMAGE_PROTOCOL`:

```sh
COLI_IMAGE_PROTOCOL=sixel coli chat --model ...    # kitty | iterm | sixel | blocks | none
```

`/render` switches it inside a session. Other knobs: `COLI_IMAGE_PREVIEW=0`
turns the live preview off, and `COLI_IMAGE_MAX_PX` caps the width of a sixel
image in pixels (default 1024).

## HTTP API

`coli serve --model ~/Models/Qwen-Image-2.1` starts the gateway on
`127.0.0.1:8000`, with the same API key, CORS and Host rules as for text
models (`--api-key` or `COLI_API_KEY`).

```sh
curl -s http://127.0.0.1:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen-image-2.1-colibri", "prompt": "a red fox in the snow",
       "size": "1024x576", "steps": 8, "seed": 42}' \
  | python3 -c 'import sys, json, base64; d = json.load(sys.stdin);
open("fox.png", "wb").write(base64.b64decode(d["data"][0]["b64_json"])); print(d["colibri"])'
```

Request fields: `model`, `prompt`, `size` (`"WxH"`, or `width` and
`height`), `steps`, `seed`, `n` (must be 1), `response_format` (only
`b64_json`), `stream`. The answer:

```
{"created": <unix time>,
 "data": [{"b64_json": "<PNG, base64>", "revised_prompt": null}],
 "colibri": {"width": 1024, "height": 576, "seed": 42, "steps": 8,
             "timings": {"encode": <s>, "denoise": <s>, "decode": <s>}}}
```

`colibri.seed` is the seed actually used, drawn by the server when the request
did not give one.

With `"stream": true` the answer is Server-Sent Events:

```
event: image_generation.progress
data: {"stage": "denoise", "step": 3, "steps": 8, "elapsed": <seconds so far>}

event: image_generation.partial_image
data: {"b64_json": "<small PNG>", "partial_image_index": 0}

event: image_generation.completed
data: {"created": ..., "data": [...], "colibri": {...}}

data: [DONE]
```

`step` counts the denoising steps already completed (0 as the first one
starts). Partial images are small previews and arrive only when the engine
sends them; `"partial_images": 0` turns them off and `"partial_images": k`
spreads at most k of them over the run. If the generation fails after the
stream has started, the stream ends with `event: error` carrying
`{"error": {"message": ..., "type": ...}}` and then `data: [DONE]`.
Closing the connection cancels the generation.

Errors are OpenAI-shaped: HTTP 400 for a bad size, an empty prompt or
`n` other than 1 (the message names the rule); 503 when the queue is full,
since the engine draws one image at a time. `GET /v1/models` marks the model
with `"capabilities": ["image_generation"]` and its size rules;
`/v1/chat/completions` answers 400 and points to the images endpoint.

## Timings

TBD (being measured).

## License

Qwen-Image-2.1 is released under the **Qwen Research License**: research and
other non-commercial use only. Read `LICENSE` in the checkpoint folder before
using the images or the model for anything else, and never redistribute the
weights. colibri's code is under its own license; the weights are not.
