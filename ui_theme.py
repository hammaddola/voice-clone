"""Dark "studio" look for the Gradio app: palette, type and component styling, plus small HTML builders.

Palette: UI/UX Pro Max "violet + cyan on dark"; cyan darkened to #0E7490 so white button text passes 4.5:1.
"""
import html
import inspect

import gradio as gr

BG, CARD, CARD2, LINE = "#0B1020", "#141B2D", "#1A2338", "rgba(255,255,255,0.09)"
FG, MUTED = "#F1F5F9", "#9AA7BD"
VIOLET, CYAN, OK, WARN, BAD = "#7C3AED", "#0E7490", "#34D399", "#FBBF24", "#F87171"
GRAD = f"linear-gradient(135deg, {VIOLET}, {CYAN})"


def _both(**kw):
    """Same value for light and dark, so the app is always dark regardless of the OS theme."""
    params = inspect.signature(gr.themes.Base.set).parameters
    return {**kw, **{f"{k}_dark": v for k, v in kw.items() if f"{k}_dark" in params}}


THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"],
    radius_size=gr.themes.sizes.radius_lg,
).set(**_both(
    body_background_fill=BG,
    body_text_color=FG,
    body_text_color_subdued=MUTED,
    background_fill_primary=CARD,
    background_fill_secondary=CARD2,
    block_background_fill=CARD,
    block_border_color=LINE,
    block_border_width="1px",
    block_label_background_fill=CARD2,
    block_label_text_color=MUTED,
    block_title_text_color=MUTED,
    block_radius="16px",
    block_shadow="none",
    border_color_primary=LINE,
    border_color_accent=VIOLET,
    input_background_fill=BG,
    input_border_color="rgba(255,255,255,0.14)",
    input_border_color_focus=VIOLET,
    input_placeholder_color="#7C879C",
    input_radius="12px",
    checkbox_background_color=BG,
    checkbox_background_color_selected=VIOLET,
    checkbox_border_color="rgba(255,255,255,0.25)",
    checkbox_label_background_fill=CARD2,
    checkbox_label_background_fill_selected=CARD2,
    slider_color=VIOLET,
    color_accent=VIOLET,
    color_accent_soft="rgba(124,58,237,0.18)",
    button_primary_background_fill=GRAD,
    button_primary_background_fill_hover=GRAD,
    button_primary_text_color="#FFFFFF",
    button_primary_border_color="transparent",
    button_secondary_background_fill=CARD2,
    button_secondary_background_fill_hover="#222C45",
    button_secondary_text_color=FG,
    button_secondary_border_color="rgba(255,255,255,0.14)",
    button_cancel_background_fill="transparent",
    button_cancel_text_color=BAD,
    button_cancel_border_color="rgba(248,113,113,0.45)",
    button_large_radius="12px",
    panel_background_fill=CARD,
    table_even_background_fill=CARD,
    table_odd_background_fill=CARD2,
))

HEAD = '<meta name="color-scheme" content="dark">'

CSS = f"""
:root {{ --vc-grad: {GRAD}; --vc-muted: {MUTED}; --vc-line: {LINE}; --vc-card2: {CARD2}; }}
body {{ background: {BG} radial-gradient(1100px 520px at 85% -10%, rgba(124,58,237,.16), transparent 60%) fixed !important; }}
.gradio-container {{ width: 100% !important; max-width: 1320px !important; margin: 0 auto !important; background: transparent !important; }}
.vc-card .block {{ border-color: rgba(255,255,255,.06) !important; background: rgba(255,255,255,.015) !important; }}
.block:has(.vc-step, .vc-hint, .vc-empty, .vc-stats, .vc-verdict, .vc-header) {{ background: transparent !important;
  border: 0 !important; padding: 0 !important; box-shadow: none !important; }}
footer {{ display: none !important; }}

/* header */
.vc-header {{ display: flex; align-items: center; gap: 14px; padding: 6px 2px 4px; }}
.vc-mark {{ width: 42px; height: 42px; border-radius: 12px; background: var(--vc-grad); display: grid; place-items: center;
  box-shadow: 0 8px 24px rgba(124,58,237,.35); flex: none; }}
.vc-title {{ font-size: 20px; font-weight: 650; letter-spacing: .01em; margin: 0; }}
.vc-sub {{ color: var(--vc-muted); font-size: 13.5px; margin: 2px 0 0; }}
.vc-pills {{ margin-left: auto; display: flex; gap: 8px; flex-wrap: wrap; }}
.vc-pill {{ display: inline-flex; align-items: center; gap: 7px; padding: 6px 12px; border-radius: 999px;
  border: 1px solid var(--vc-line); color: var(--vc-muted); font-size: 12.5px; background: rgba(255,255,255,.02); }}
.vc-dot {{ width: 7px; height: 7px; border-radius: 50%; background: {OK}; box-shadow: 0 0 0 3px rgba(52,211,153,.15); }}

/* tabs as a segmented pill bar */
div[role=tablist] {{ gap: 4px; padding: 5px; border: 1px solid var(--vc-line) !important; border-radius: 14px;
  background: {CARD}; width: max-content; max-width: 100%; overflow-x: auto; margin-bottom: 14px; }}
button[role=tab] {{ border-radius: 10px !important; padding: 9px 16px !important; font-weight: 550 !important;
  color: var(--vc-muted) !important; border: 0 !important; min-height: 40px; cursor: pointer; transition: background .2s, color .2s; }}
button[role=tab][aria-selected=true] {{ background: var(--vc-card2) !important; color: {FG} !important;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.12), 0 4px 14px rgba(0,0,0,.3); }}
button[role=tab][aria-selected=true]::after {{ display: none !important; }}

/* cards + step headings */
.vc-card {{ background: {CARD}; border: 1px solid var(--vc-line); border-radius: 18px; padding: 18px !important; gap: 12px !important; }}
.vc-step {{ display: flex; align-items: center; gap: 10px; font-size: 12.5px; font-weight: 650; letter-spacing: .09em;
  text-transform: uppercase; color: var(--vc-muted); margin: 0 0 2px; }}
.vc-step b {{ width: 24px; height: 24px; border-radius: 8px; display: grid; place-items: center; background: var(--vc-card2);
  border: 1px solid rgba(255,255,255,.12); color: {FG}; font-size: 12px; letter-spacing: 0; }}
.vc-hint {{ color: var(--vc-muted); font-size: 13px; line-height: 1.55; }}

/* consent: green when ticked */
.vc-consent {{ border: 1px solid rgba(52,211,153,.30) !important; background: rgba(52,211,153,.06) !important; border-radius: 12px !important; }}

/* buttons */
button.primary {{ box-shadow: 0 10px 28px rgba(124,58,237,.30); font-weight: 650 !important; min-height: 48px; cursor: pointer;
  transition: transform .15s ease, box-shadow .2s ease, filter .2s ease; }}
button.primary:hover {{ filter: brightness(1.08); box-shadow: 0 12px 32px rgba(124,58,237,.42); }}
button.primary:active {{ transform: translateY(1px); }}
button {{ cursor: pointer; }}
button.lg {{ min-height: 48px; }}
:focus-visible {{ outline: 2px solid #A78BFA !important; outline-offset: 2px !important; }}

/* results */
.vc-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}
.vc-stat {{ display: flex; gap: 12px; align-items: center; padding: 14px; border-radius: 14px; background: var(--vc-card2);
  border: 1px solid var(--vc-line); }}
.vc-stat strong {{ display: block; font-size: 20px; font-weight: 650; line-height: 1.2; }}
.vc-stat span {{ color: var(--vc-muted); font-size: 12.5px; }}
.vc-ic {{ width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center; flex: none; }}
.vc-note {{ display: flex; gap: 10px; align-items: flex-start; margin-top: 10px; padding: 11px 14px; border-radius: 12px; font-size: 13.5px; }}
.vc-note.warn {{ background: rgba(251,191,36,.08); border: 1px solid rgba(251,191,36,.28); color: #FCD9A0; }}
.vc-verdict {{ display: flex; gap: 16px; align-items: center; padding: 18px; border-radius: 16px; }}
.vc-verdict strong {{ font-size: 18px; }} .vc-verdict p {{ margin: 2px 0 0; color: var(--vc-muted); }}
.vc-bits {{ display: grid; grid-template-columns: repeat(16, 1fr); gap: 4px; margin-top: 12px; }}
.vc-bits i {{ font: 600 12px/26px "JetBrains Mono", Consolas, monospace; text-align: center; font-style: normal; border-radius: 6px; }}
.vc-empty {{ color: var(--vc-muted); font-size: 13.5px; padding: 6px 2px; }}

@media (max-width: 720px) {{ .vc-pills {{ display: none; }} .vc-bits {{ grid-template-columns: repeat(8, 1fr); }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
"""

_ICONS = {  # Lucide paths (MIT)
    "wave": '<path d="M2 10v3M6 6v11M10 3v18M14 8v7M18 5v13M22 10v3"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4M12 17h.01"/>',
    "x": '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6M9 9l6 6"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6" rx="1"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',
    "activity": '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>',
}


def icon(name, size=18, color="currentColor"):
    return (f'<svg aria-hidden="true" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{_ICONS[name]}</svg>')


def header(device: str) -> str:
    return f"""<div class="vc-header">
  <div class="vc-mark">{icon("wave", 22, "#fff")}</div>
  <div><h1 class="vc-title">Voice Clone</h1>
       <p class="vc-sub">Zero-shot multilingual voice cloning. Only clone voices you have permission to use.</p></div>
  <div class="vc-pills"><span class="vc-pill"><span class="vc-dot"></span>{icon("cpu", 14)}Running on {html.escape(device.upper())}</span>
       <span class="vc-pill">{icon("shield", 14)}Every output watermarked</span></div>
</div>"""


def step(n, title) -> str:
    return f'<p class="vc-step"><b>{n}</b>{html.escape(title)}</p>'


def _ring(pct):
    c = 125.66  # 2πr for r=20
    return (f'<svg aria-hidden="true" width="48" height="48" viewBox="0 0 48 48"><defs><linearGradient id="vcg">'
            f'<stop offset="0" stop-color="#8B5CF6"/><stop offset="1" stop-color="#22D3EE"/></linearGradient></defs>'
            f'<circle cx="24" cy="24" r="20" stroke="#26304A" stroke-width="6" fill="none"/>'
            f'<circle cx="24" cy="24" r="20" stroke="url(#vcg)" stroke-width="6" fill="none" stroke-linecap="round" '
            f'stroke-dasharray="{c * pct / 100:.1f} {c:.1f}" transform="rotate(-90 24 24)"/></svg>')


def notes(warnings) -> str:
    return "".join(f'<div class="vc-note warn" role="status">{icon("alert", 16)}<span>{html.escape(w)}</span></div>'
                   for w in warnings)


def result_card(similarity: float, detail: str, warnings=()) -> str:
    return f"""<div class="vc-stats">
  <div class="vc-stat">{_ring(similarity)}<div><strong>{similarity:.1f}%</strong><span>Speaker similarity</span></div></div>
  <div class="vc-stat"><span class="vc-ic" style="background:rgba(52,211,153,.12);color:{OK}">{icon("shield")}</span>
       <div><strong>Embedded</strong><span>AudioSeal watermark</span></div></div>
  <div class="vc-stat"><span class="vc-ic" style="background:rgba(34,211,238,.12);color:#22D3EE">{icon("activity")}</span>
       <div><strong>{html.escape(detail)}</strong><span>Engine</span></div></div>
</div>{notes(warnings)}"""


def watermark_card(r: dict, app_id) -> str:
    if r["is_this_app"]:
        tone, ic, title, sub = OK, "shield", "Contains this app's watermark", "This audio was generated by Voice Clone."
    elif r["watermarked_by_someone"]:
        tone, ic, title, sub = WARN, "alert", "Watermarked, but not by this app", "An AudioSeal watermark with a different ID."
    else:
        tone, ic, title, sub = BAD, "x", "No watermark detected", "Not produced by this app, or heavily altered."
    bits = "".join(f'<i style="background:{"rgba(52,211,153,.16)" if b else "rgba(255,255,255,.04)"};'
                   f'border:1px solid {"rgba(52,211,153,.4)" if b else "rgba(255,255,255,.1)"}">{int(b)}</i>' for b in app_id)
    return f"""<div class="vc-verdict" style="border:1px solid {tone}55;background:{tone}12" role="status">
  <span class="vc-ic" style="width:56px;height:56px;border-radius:18px;background:{tone}22;color:{tone}">{icon(ic, 28)}</span>
  <div><strong>{title}</strong><p>{sub}</p></div></div>
<div class="vc-stats" style="margin-top:12px">
  <div class="vc-stat"><div><strong>{r["probability"]:.2f}</strong><span>Detection probability</span></div></div>
  <div class="vc-stat"><div><strong>{r["bit_errors"]} / 16</strong><span>App-ID bit errors</span></div></div>
</div>
<p class="vc-hint" style="margin-top:12px">This app's 16-bit ID</p><div class="vc-bits">{bits}</div>"""


EMPTY_RESULT = '<p class="vc-empty">Your result appears here: similarity score, watermark status and any quality warnings.</p>'
