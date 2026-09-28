import logging
import math
import os
from pathlib import Path

import numpy as np
import svgwrite

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import drawing
import lyrics
from rnn import rnn

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FONT_SIZE = 36.0
MIN_FONT_SIZE = 18.0
MAX_FONT_SIZE = 72.0


class Hand:

    def __init__(self):
        self.nn = rnn(
            log_dir=str(BASE_DIR / 'logs'),
            checkpoint_dir=str(BASE_DIR / 'checkpoints'),
            prediction_dir=str(BASE_DIR / 'predictions'),
            learning_rates=[.0001, .00005, .00002],
            batch_sizes=[32, 64, 64],
            patiences=[1500, 1000, 500],
            beta1_decays=[.9, .9, .9],
            validation_batch_size=32,
            optimizer='rms',
            num_training_steps=100000,
            warm_start_init_step=17900,
            regularization_constant=0.0,
            keep_prob=1.0,
            enable_parameter_averaging=False,
            min_steps_to_checkpoint=2000,
            log_interval=20,
            logging_level=logging.CRITICAL,
            grad_clip=10,
            lstm_size=400,
            output_mixture_components=20,
            attention_mixture_components=10
        )
        self.nn.restore()

    def write(
        self,
        filename,
        lines,
        biases=None,
        styles=None,
        stroke_colors=None,
        stroke_widths=None,
        alignment="center",
        font_size=DEFAULT_FONT_SIZE,
    ):
        document = self.render(
            lines=lines,
            biases=biases,
            styles=styles,
            stroke_colors=stroke_colors,
            stroke_widths=stroke_widths,
            alignment=alignment,
            font_size=font_size,
        )
        self._draw(document, filename)

    def render(
        self,
        lines,
        biases=None,
        styles=None,
        stroke_colors=None,
        stroke_widths=None,
        alignment="center",
        font_size=DEFAULT_FONT_SIZE,
    ):
        """Render handwriting into a platform-neutral vector document."""

        valid_char_set = set(drawing.alphabet)
        if alignment not in {"left", "center"}:
            raise ValueError("Alignment must be either 'left' or 'center'.")
        if isinstance(font_size, bool):
            raise ValueError("Font size must be a number between 18 and 72.")
        try:
            font_size = float(font_size)
        except (TypeError, ValueError):
            raise ValueError("Font size must be a number between 18 and 72.") from None
        if not math.isfinite(font_size) or not MIN_FONT_SIZE <= font_size <= MAX_FONT_SIZE:
            raise ValueError("Font size must be a number between 18 and 72.")
        for line_num, line in enumerate(lines):
            if len(line) > 75:
                raise ValueError(
                    "Each line must be at most 75 characters. "
                    f"Line {line_num} contains {len(line)}"
                )

            for char in line:
                if char not in valid_char_set:
                    raise ValueError(
                        f"Invalid character {char} detected in line {line_num}. "
                        f"Valid character set is {valid_char_set}"
                    )

        strokes = self._sample(lines, biases=biases, styles=styles)
        return self._layout(
            strokes,
            lines,
            stroke_colors=stroke_colors,
            stroke_widths=stroke_widths,
            alignment=alignment,
            font_size=font_size,
        )

    def _sample(self, lines, biases=None, styles=None):
        num_samples = len(lines)
        max_tsteps = 40*max([len(i) for i in lines])
        biases = biases if biases is not None else [0.5]*num_samples

        x_prime = np.zeros([num_samples, 1200, 3])
        x_prime_len = np.zeros([num_samples])
        chars = np.zeros([num_samples, 120])
        chars_len = np.zeros([num_samples])

        if styles is not None:
            for i, (cs, style) in enumerate(zip(lines, styles, strict=True)):
                x_p = np.load(BASE_DIR / f'styles/style-{style}-strokes.npy')
                c_p = np.load(BASE_DIR / f'styles/style-{style}-chars.npy').tobytes().decode('utf-8')

                c_p = str(c_p) + " " + cs
                c_p = drawing.encode_ascii(c_p)
                c_p = np.array(c_p)

                x_prime[i, :len(x_p), :] = x_p
                x_prime_len[i] = len(x_p)
                chars[i, :len(c_p)] = c_p
                chars_len[i] = len(c_p)

        else:
            for i in range(num_samples):
                encoded = drawing.encode_ascii(lines[i])
                chars[i, :len(encoded)] = encoded
                chars_len[i] = len(encoded)

        [samples] = self.nn.session.run(
            [self.nn.sampled_sequence],
            feed_dict={
                self.nn.prime: styles is not None,
                self.nn.x_prime: x_prime,
                self.nn.x_prime_len: x_prime_len,
                self.nn.num_samples: num_samples,
                self.nn.sample_tsteps: max_tsteps,
                self.nn.c: chars,
                self.nn.c_len: chars_len,
                self.nn.bias: biases
            }
        )
        samples = [sample[~np.all(sample == 0.0, axis=1)] for sample in samples]
        return samples

    def _layout(
        self,
        strokes,
        lines,
        stroke_colors=None,
        stroke_widths=None,
        alignment="center",
        font_size=DEFAULT_FONT_SIZE,
    ):
        stroke_colors = stroke_colors or ['black']*len(lines)
        stroke_widths = stroke_widths or [2]*len(lines)

        font_scale = font_size / DEFAULT_FONT_SIZE
        line_height = 60 * font_scale
        view_width = 1000
        view_height = line_height*(len(strokes) + 1)
        paths = []

        initial_coord = np.array([0, -(3*line_height / 4)])
        for offsets, line, color, width in zip(
            strokes, lines, stroke_colors, stroke_widths, strict=True
        ):

            if not line:
                initial_coord[1] -= line_height
                continue

            offsets = offsets.copy()
            offsets[:, :2] *= 1.5 * font_scale
            strokes = drawing.offsets_to_coords(offsets)
            strokes = drawing.denoise(strokes)
            strokes[:, :2] = drawing.align(strokes[:, :2])

            strokes[:, 1] *= -1
            strokes[:, :2] -= strokes[:, :2].min() + initial_coord
            if alignment == "center":
                strokes[:, 0] += (view_width - strokes[:, 0].max()) / 2
            else:
                left_padding = 60
                strokes[:, 0] += left_padding

            prev_eos = 1.0
            points = []
            for x, y, eos in zip(*strokes.T, strict=True):
                points.append(
                    {
                        "x": float(x),
                        "y": float(y),
                        "move": bool(prev_eos == 1.0),
                    }
                )
                prev_eos = eos
            paths.append(
                {
                    "strokeColor": color,
                    "lineWidth": float(width) * font_scale,
                    "points": points,
                }
            )

            initial_coord[1] -= line_height

        return {
            "width": float(view_width),
            "height": float(view_height),
            "backgroundColor": "#FFFFFF",
            "paths": paths,
        }

    def _draw(self, document, filename):
        """Write a vector document to SVG."""

        dwg = svgwrite.Drawing(filename=str(filename))
        dwg.viewbox(width=document["width"], height=document["height"])
        dwg.add(
            dwg.rect(
                insert=(0, 0),
                size=(document["width"], document["height"]),
                fill=document["backgroundColor"],
            )
        )

        for rendered_path in document["paths"]:
            commands = []
            for point in rendered_path["points"]:
                command = "M" if point["move"] else "L"
                commands.append(f'{command}{point["x"]},{point["y"]}')

            path = svgwrite.path.Path(" ".join(commands))
            path = path.stroke(
                color=rendered_path["strokeColor"],
                width=rendered_path["lineWidth"],
                linecap="round",
            ).fill("none")
            dwg.add(path)

        dwg.save()


if __name__ == '__main__':
    hand = Hand()

    # usage demo
    lines = [
        "Now this is a story all about how",
        "My life got flipped turned upside down",
        "And I'd like to take a minute, just sit right there",
        "I'll tell you how I became the prince of a town called Bel-Air",
    ]
    biases = [.75 for i in lines]
    styles = [9 for i in lines]
    stroke_colors = ['red', 'green', 'black', 'blue']
    stroke_widths = [1, 2, 1, 2]

    hand.write(
        filename='img/usage_demo.svg',
        lines=lines,
        biases=biases,
        styles=styles,
        stroke_colors=stroke_colors,
        stroke_widths=stroke_widths
    )

    # demo number 1 - fixed bias, fixed style
    lines = lyrics.all_star.split("\n")
    biases = [.75 for i in lines]
    styles = [12 for i in lines]

    hand.write(
        filename='img/all_star.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )

    # demo number 2 - fixed bias, varying style
    lines = lyrics.downtown.split("\n")
    biases = [.75 for i in lines]
    styles = np.cumsum(np.array([len(i) for i in lines]) == 0).astype(int)

    hand.write(
        filename='img/downtown.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )

    # demo number 3 - varying bias, fixed style
    lines = lyrics.give_up.split("\n")
    biases = .2*np.flip(np.cumsum([len(i) == 0 for i in lines]), 0)
    styles = [7 for i in lines]

    hand.write(
        filename='img/give_up.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )
