"""Render evidence charts with the bundled ReportLab/PyMuPDF Python."""
import csv
import hashlib
import json
from pathlib import Path
import sys

from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib.colors import HexColor
import pymupdf

HERE = Path(__file__).resolve().parent
GROUPS = ('A', 'B', 'C', 'D', 'F', 'U')
SEEDS = (10, 11, 12)
COLORS = dict(zip(GROUPS, (HexColor('#2563eb'), HexColor('#7c3aed'),
                            HexColor('#dc2626'), HexColor('#ea580c'),
                            HexColor('#0f766e'), HexColor('#64748b'))))
INK, MUTED, GRID = HexColor('#172033'), HexColor('#4b5563'), HexColor('#e5e7eb')


def read(path):
    if not path.exists():
        return []
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def number(value):
    if value in (None, '', 'None', 'null'):
        return None
    return float(value)


def label(drawing, x, y, string, size=10, color=INK, bold=False):
    drawing.add(String(x, y, string, fontName='Helvetica-Bold' if bold else 'Helvetica',
                       fontSize=size, fillColor=color))


def save(drawing, folder, stem):
    svg, png = folder / (stem + '.svg'), folder / (stem + '.png')
    renderSVG.drawToFile(drawing, str(svg))
    with pymupdf.open(stream=renderPDF.drawToString(drawing), filetype='pdf') as pdf:
        pdf[0].get_pixmap(matrix=pymupdf.Matrix(1.6, 1.6), alpha=False).save(str(png))
    return svg, png


def search_chart(rows):
    drawing = Drawing(1210, 465)
    drawing.add(Rect(0, 0, 1210, 465, fillColor=HexColor('#ffffff'), strokeColor=None))
    label(drawing, 40, 430, 'i2c search: best feasible LUT vs 1,000 action candidates', 18, bold=True)
    label(drawing, 40, 408, 'Three training seeds; candidate 0 is initial mapping (not charged to budget).', 10, MUTED)
    for index, group in enumerate(GROUPS):
        x = 45 + index * 191
        drawing.add(Line(x, 383, x + 26, 383, strokeColor=COLORS[group], strokeWidth=3))
        label(drawing, x + 33, 379, group + ('  H50/K10' if group == 'D' else ''), 10)
    valid = []
    for row in rows:
        value = number(row.get('best_feasible_luts', row.get('luts')))
        if value is not None and row.get('feasible', 'True') == 'True':
            valid.append(value)
    low = int(min(valid) - 3) if valid else 295
    high = int(max(valid) + 3) if valid else 365
    if high - low < 8:
        low -= 4
        high += 4
    for slot, seed in enumerate(SEEDS):
        left = 45 + slot * 399
        label(drawing, left + 10, 350, 'Training seed ' + str(seed), 13, bold=True)
        plot = LinePlot()
        plot.x, plot.y, plot.width, plot.height = left + 20, 77, 350, 235
        plot.xValueAxis.valueMin, plot.xValueAxis.valueMax = 0, 1000
        plot.xValueAxis.valueSteps = [0, 250, 500, 750, 1000]
        plot.yValueAxis.valueMin, plot.yValueAxis.valueMax = low, high
        plot.yValueAxis.valueSteps = [round(low + (high - low) * k / 4) for k in range(5)]
        plot.yValueAxis.visibleGrid = True
        plot.yValueAxis.gridStrokeColor = GRID
        plotted = []
        for group in GROUPS:
            points = sorted((int(row['candidate']), number(row.get('best_feasible_luts', row.get('luts'))))
                            for row in rows if row.get('group') == group and
                            str(row.get('training_seed', row.get('seed'))) == str(seed) and
                            number(row.get('best_feasible_luts', row.get('luts'))) is not None and
                            row.get('feasible', 'True') == 'True')
            if points:
                plotted.append((group, points))
        if plotted:
            plot.data = [points for _, points in plotted]
            for series, (group, _) in enumerate(plotted):
                plot.lines[series].strokeColor = COLORS[group]
                plot.lines[series].strokeWidth = 1.9 if group == 'D' else 1.2
                if group in ('F', 'U'):
                    plot.lines[series].strokeDashArray = [4, 3]
            drawing.add(plot)
        else:
            label(drawing, left + 110, 190, 'No verified feasible observations', 9, MUTED)
        label(drawing, left + 143, 45, 'Action candidates', 10, MUTED)
    return drawing


def evaluation_chart(rows):
    drawing = Drawing(1210, 495)
    drawing.add(Rect(0, 0, 1210, 495, fillColor=HexColor('#ffffff'), strokeColor=None))
    label(drawing, 40, 455, 'i2c fixed-policy evaluation: model means, 30 physical rollouts each', 17, bold=True)
    label(drawing, 40, 433, 'Dots are training models. U is one shared random bank; 10/50 prefixes are dependent.', 10, MUTED)
    valid = [number(r.get('mean_best_feasible_luts', r.get('mean_luts'))) for r in rows
             if number(r.get('mean_best_feasible_luts', r.get('mean_luts'))) is not None]
    low = int(min(valid) - 3) if valid else 295
    high = int(max(valid) + 3) if valid else 365
    if high - low < 10:
        low -= 5
        high += 5
    for col, prefix in enumerate((10, 50)):
        left = 45 + col * 605
        label(drawing, left + 170, 394, f'Best feasible LUT in {prefix} steps', 13, bold=True)
        x0, x1, y0, y1 = left + 52, left + 550, 112, 354
        drawing.add(Line(x0, y0, x0, y1, strokeColor=MUTED))
        drawing.add(Line(x0, y0, x1, y0, strokeColor=MUTED))
        for tick in range(5):
            value = low + (high - low) * tick / 4
            y = y0 + (y1 - y0) * tick / 4
            drawing.add(Line(x0, y, x1, y, strokeColor=GRID))
            label(drawing, x0 - 36, y - 3, f'{value:.0f}', 9, MUTED)
        for index, group in enumerate(GROUPS):
            x = x0 + 45 + index * 82
            label(drawing, x - 4, 90, group, 11, COLORS[group], bold=True)
            selected = [(str(r.get('training_seed', r.get('seed'))), number(r.get('mean_best_feasible_luts', r.get('mean_luts'))))
                        for r in rows if r.get('group') == group and int(r.get('prefix', 0)) == prefix and
                        number(r.get('mean_best_feasible_luts', r.get('mean_luts'))) is not None and
                        str(r.get('complete', 'True')) == 'True']
            if selected:
                for model, value in selected:
                    jitter = {str(SEEDS[0]): -8, str(SEEDS[1]): 0, str(SEEDS[2]): 8}.get(model, 0)
                    y = y0 + (value - low) * (y1 - y0) / (high - low)
                    drawing.add(Circle(x + jitter, y, 5, strokeColor=COLORS[group],
                                       fillColor=COLORS[group]))
                mean = sum(value for _, value in selected) / len(selected)
                y = y0 + (mean - low) * (y1 - y0) / (high - low)
                drawing.add(Line(x - 15, y, x + 15, y, strokeColor=INK, strokeWidth=1.7))
            else:
                label(drawing, x - 5, 192, '—', 12, MUTED)
        if prefix == 50:
            label(drawing, left + 145, 53, 'A/B: beyond training horizon; U: one shared bank.', 10, MUTED)
        else:
            label(drawing, left + 165, 53, 'A/B: within training horizon.', 10, MUTED)
    return drawing


def render(folder=HERE):
    folder = Path(folder)
    curves = read(folder / 'search-curves.csv')
    means = read(folder / 'evaluation-means.csv')
    outputs = list(save(search_chart(curves), folder, 'search-curves'))
    outputs += list(save(evaluation_chart(means), folder, 'evaluation-comparison'))
    paths = [folder / 'search-curves.csv', folder / 'evaluation-means.csv', Path(__file__), *outputs]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    record = dict(command=[sys.executable, '-B', str(Path(__file__))],
        source=hashes.get(Path(__file__).name), source_training_rows=len(curves),
        source_evaluation_bank_rows=len(means), hashes=hashes,
        status='rendered' if curves and means else 'incomplete_inputs')
    (folder / 'plot-validation.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    render()
