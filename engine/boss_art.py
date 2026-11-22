"""Placeholder art for Count Aggregor, his Tally Bats and NULL (same palette as the others)."""

from __future__ import annotations

import math

import pygame

from engine import render



def draw_barracks(canvas: pygame.Surface, t: int, ground_y: int = 60) -> None:
    """A long stone hall where bats hang in tidy rows, with a tally board on the wall."""
    pygame.draw.rect(canvas, render.INK, (0, 0, 320, ground_y))
    pygame.draw.rect(canvas, render.NIGHT, (0, 8, 320, ground_y - 8))
    for x in range(0, 320, 40):
        pygame.draw.rect(canvas, render.DUSK, (x + 4, 14, 32, ground_y - 14))
        pygame.draw.circle(canvas, render.DUSK, (x + 20, 14), 16)
        pygame.draw.rect(canvas, render.INK, (x + 10, 20, 20, ground_y - 20))
        pygame.draw.circle(canvas, render.INK, (x + 20, 20), 10)
    pygame.draw.rect(canvas, render.BARK, (0, 6, 320, 3))
    for i, x in enumerate(range(8, 320, 14)):
        if (x // 14) % 3 == 1:
            continue
        sway = 1 if (t // 400 + i) % 4 == 0 else 0
        pygame.draw.polygon(canvas, render.SLATE, [(x - 2, 9), (x + 2, 9), (x + sway, 14)])
    board = pygame.Rect(120, 18, 40, 22)
    pygame.draw.rect(canvas, render.BARK, board)
    pygame.draw.rect(canvas, render.LEATHER, board, 1)
    for g in range(3):
        gx = board.x + 4 + g * 12
        for k in range(4):
            pygame.draw.line(canvas, render.CREAM, (gx + k * 2, board.y + 5), (gx + k * 2, board.y + 15))
        pygame.draw.line(canvas, render.CREAM, (gx - 1, board.y + 13), (gx + 8, board.y + 7))
    for x in (60, 260):
        pygame.draw.rect(canvas, render.CREAM, (x - 1, 30, 3, 8))
        flame = render.GOLD if (t // 130 + x) % 3 else render.BUTTER
        pygame.draw.polygon(canvas, flame, [(x - 2, 30), (x + 3, 30), (x + 1, 25 - (t // 90) % 2)])
    pygame.draw.rect(canvas, render.SLATE, (0, ground_y, 320, 180 - ground_y))
    for y in range(ground_y + 6, 180, 8):
        pygame.draw.line(canvas, render.DUSK, (0, y), (320, y))
        for x in range((y // 8 % 2) * 10, 320, 20):
            pygame.draw.line(canvas, render.DUSK, (x, y - 7), (x, y))


def draw_count(canvas: pygame.Surface, centre: tuple[float, float], t: int, flash: bool,
               fainted: bool, shielded: bool, moving: bool) -> None:
    """Placeholder Count Aggregor: a tall caped figure with an abacus. While the quorum holds,
    a ring of tally marks circles him."""
    cx, cy = render.snap(centre[0]), render.snap(centre[1])
    pygame.draw.ellipse(canvas, render.INK, (cx - 13, cy + 16, 26, 5))
    if fainted:
        pygame.draw.rect(canvas, render.WINE, (cx - 16, cy + 11, 32, 6))
        pygame.draw.rect(canvas, render.PARCHMENT, (cx - 4, cy + 10, 8, 4))
        return
    cape, coat, skin = ((render.WHITE,) * 3 if flash else (render.WINE, render.INK, render.PARCHMENT))
    bob = 1 if (t // 200) % 2 else 0
    pygame.draw.polygon(canvas, cape, [(cx - 12, cy + 17), (cx - 7, cy - 8 + bob),
                                       (cx + 7, cy - 8 + bob), (cx + 12, cy + 17)])
    pygame.draw.rect(canvas, coat, (cx - 6, cy - 7 + bob, 12, 23))
    pygame.draw.polygon(canvas, cape, [(cx - 10, cy - 10 + bob), (cx - 3, cy - 6 + bob),
                                       (cx - 6, cy - 16 + bob)])
    pygame.draw.polygon(canvas, cape, [(cx + 10, cy - 10 + bob), (cx + 3, cy - 6 + bob),
                                       (cx + 6, cy - 16 + bob)])
    pygame.draw.rect(canvas, skin, (cx - 4, cy - 16 + bob, 8, 9))
    if not flash:
        pygame.draw.rect(canvas, render.RED, (cx - 3, cy - 13 + bob, 2, 1))
        pygame.draw.rect(canvas, render.RED, (cx + 1, cy - 13 + bob, 2, 1))
        pygame.draw.rect(canvas, render.INK, (cx - 4, cy - 17 + bob, 8, 2))
        pygame.draw.polygon(canvas, render.INK, [(cx - 1, cy - 15 + bob), (cx + 1, cy - 15 + bob),
                                                 (cx, cy - 13 + bob)])
    frame = pygame.Rect(cx + 7, cy - 2 + bob, 9, 10)
    pygame.draw.rect(canvas, render.LEATHER if not flash else render.WHITE, frame, 1)
    for row in range(3):
        y = frame.y + 2 + row * 3
        pygame.draw.line(canvas, render.SAND, (frame.x + 1, y), (frame.right - 2, y))
        slide = (t // 300 + row) % 3
        canvas.set_at((frame.x + 2 + slide, y), render.GOLD)
        canvas.set_at((frame.x + 5, y), render.GOLD)
    if moving and (t // 140) % 2:
        pygame.draw.line(canvas, cape, (cx - 12, cy + 17), (cx - 15, cy + 13))
    if shielded:
        spin = t / 400
        for i in range(10):
            angle = i / 10 * math.tau + spin
            x = render.snap(cx + math.cos(angle) * 20)
            y = render.snap(cy + math.sin(angle) * 22)
            colour = render.GOLD if i % 5 == 4 else render.CREAM
            pygame.draw.line(canvas, colour, (x, y - 2), (x, y + 1))


def draw_bat(canvas: pygame.Surface, pos: tuple[float, float], t: int, phase: float,
             flash: bool = False, state: str = "hover") -> None:
    """One Tally Bat. Diving bats flap fast with bright eyes, fleeing ones go grey."""
    x, y = render.snap(pos[0]), render.snap(pos[1])
    body = render.WHITE if flash else (render.SLATE if state == "flee" else render.DUSK)
    edge = render.WHITE if flash else render.INK
    speed = 40 if state == "dive" else 70
    up = math.sin(t / speed + phase) > 0
    wing_y = y - 3 if up else y + 1
    pygame.draw.polygon(canvas, edge, [(x - 1, y - 1), (x - 6, wing_y), (x - 4, y + 1)])
    pygame.draw.polygon(canvas, edge, [(x + 1, y - 1), (x + 6, wing_y), (x + 4, y + 1)])
    pygame.draw.rect(canvas, body, (x - 1, y - 2, 3, 4))
    if not flash:
        eye = render.GOLD if state == "dive" else (render.GREY if state == "flee" else render.RED)
        canvas.set_at((x - 1, y - 1), eye)
        canvas.set_at((x + 1, y - 1), eye)


def draw_squad_labels(canvas: pygame.Surface, anchors: dict, counts: dict, quorum: set) -> None:
    """Each squad's name and size under it, gold while it holds the quorum."""
    f = render.font()
    for squad, spot in anchors.items():
        n = counts.get(squad, 0)
        if not n:
            continue
        text = f"{squad} {n}"
        colour = render.GOLD if squad in quorum else render.GREY
        f.draw(canvas, text, (render.snap(spot.x) - f.width(text) // 2, render.snap(spot.y) + 6),
               colour, shadow=render.INK)




def draw_null_void(canvas: pygame.Surface, t: int, ground_y: int = 60) -> None:
    """Nothing, drawn carefully: a dark field of empty cells that blink like missing values."""
    canvas.fill(render.INK)
    for row, y in enumerate(range(4, ground_y + 40, 10)):
        for col, x in enumerate(range(4 + (row % 2) * 6, 320, 24)):
            if (col * 7 + row * 3 + t // 600) % 11 == 0:
                render.font().draw(canvas, "NULL", (x, y), render.DUSK)
            elif (col + row) % 3 == 0:
                pygame.draw.rect(canvas, render.NIGHT, (x, y + 1, 16, 5), 1)
    pygame.draw.rect(canvas, render.NIGHT, (0, ground_y, 320, 180 - ground_y))
    for x in range(0, 320, 16):
        pygame.draw.line(canvas, render.DUSK, (x, ground_y), (x, 180))
    pygame.draw.line(canvas, render.VIOLET, (0, ground_y), (320, ground_y))


def draw_null(canvas: pygame.Surface, centre: tuple[float, float], t: int, flash: bool,
              fainted: bool, element: str | None, veiled: bool) -> None:
    """Placeholder NULL: Ada's own silhouette, hollow, flickering in and out of existence.
    After a Shift it takes the colour of its new element."""
    cx, cy = render.snap(centre[0]), render.snap(centre[1])
    if fainted:
        for i in range(8):
            canvas.set_at((cx - 12 + i * 3, cy + 14 + (i % 2)), render.GREY)
        return
    if not flash and (t // 90) % 17 == 0:
        return
    outline = render.WHITE if flash else render.ELEMENT_COLOURS.get(element, render.LILAC)
    jitter = 1 if (t // 110) % 5 == 0 else 0
    hat = [(cx - 9 + jitter, cy - 12), (cx + 1 + jitter, cy - 30), (cx + 10 + jitter, cy - 12)]
    pygame.draw.polygon(canvas, outline, hat, 1)
    pygame.draw.line(canvas, outline, (cx - 13, cy - 12), (cx + 14, cy - 12))
    pygame.draw.ellipse(canvas, outline, (cx - 6, cy - 12, 12, 11), 1)
    pygame.draw.polygon(canvas, outline, [(cx - 5, cy - 1), (cx - 12, cy + 17),
                                          (cx + 12, cy + 17), (cx + 5, cy - 1)], 1)
    if not flash:
        render.font().draw(canvas, "?", (cx - 1, cy + 3), outline)
        canvas.set_at((cx - 2, cy - 7), render.WHITE)
        canvas.set_at((cx + 2, cy - 7), render.WHITE)
    if veiled:
        spin = t / 300
        for i in range(12):
            if (i + int(spin * 3)) % 3 == 0:
                continue
            angle = i / 12 * math.tau - spin
            canvas.set_at((render.snap(cx + math.cos(angle) * 18),
                           render.snap(cy + math.sin(angle) * 22)), render.VIOLET)
