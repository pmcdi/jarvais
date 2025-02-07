"""
Helper functions and classes for aesthetic design.
"""

from typing import Callable
from fpdf import FPDF
from math import sqrt
from pathlib import Path
import seaborn as sns

import PIL
PIL.Image.MAX_IMAGE_PIXELS = None # Doing this to prevent PIL.Image.DecompressionBombError on large images

def config_plot(plot_type: str | None=None) -> Callable:
    """A decorator to configure plot settings using Seaborn and Matplotlib."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            from matplotlib.font_manager import fontManager, FontProperties
            script_dir = Path(__file__).resolve().parent

            # Adding unicode fonts
            font_path = (script_dir / 'fonts/Inter_28pt-Regular.ttf')
            fontManager.addfont(font_path)
            inter = FontProperties(fname=font_path)
            rc_dict = {
                "figure.dpi": 300,
                'figure.subplot.left': 0.1,
                'figure.subplot.right': 0.9,
                'figure.subplot.bottom': 0.1,
                'figure.subplot.top': 0.9
            }
            if plot_type == 'multi':
                rc_dict['figure.subplot.left'] = 0.05
                rc_dict['figure.subplot.right'] = 0.95
                rc_dict['figure.subplot.bottom'] = 0.05
                rc_dict['figure.subplot.top'] = 0.95
                rc_dict['figure.subplot.hspace'] = .15
                rc_dict['figure.subplot.wspace'] = .15
            elif plot_type == 'ft':
                rc_dict['figure.subplot.left'] = 0.22
                rc_dict['figure.subplot.right'] = 0.98

            sns.set_theme(style="darkgrid", font=inter.get_name(), rc=rc_dict)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

class PDFRounded(FPDF):
    """A subclass of FPDF that adds support for drawing rounded rectangles."""
    def rounded_rect(
        self, x: float, y: float, w: float, h: float, r: float, 
        style: str = '', corners: str = '1234'
    ) -> None:
        """Draws a rounded rectangle."""
        k = self.k
        hp = self.h
        if(style=='F'):
            op='f'
        elif(style=='FD' or style=='DF'):
            op='B'
        else:
            op='S'
        myArc = 4/3 * (sqrt(2) - 1)
        self._out('%.2F %.2F m' % ((x+r)*k,(hp-y)*k))

        xc = x+w-r
        yc = y+r
        self._out('%.2F %.2F l' % (xc*k,(hp-y)*k))
        if '2' not in corners:
            self._out('%.2F %.2F l' % ((x+w)*k,(hp-y)*k))
        else:
            self._arc(xc + r*myArc, yc - r, xc + r, yc - r*myArc, xc + r, yc)

        xc = x+w-r
        yc = y+h-r
        self._out('%.2F %.2F l' % ((x+w)*k,(hp-yc)*k))
        if '3' not in corners:
            self._out('%.2F %.2F l' % ((x+w)*k,(hp-(y+h))*k))
        else:
            self._arc(xc + r, yc + r*myArc, xc + r*myArc, yc + r, xc, yc + r)

        xc = x+r
        yc = y+h-r
        self._out('%.2F %.2F l' % (xc*k,(hp-(y+h))*k))
        if '4' not in corners:
            self._out('%.2F %.2F l' % (x*k,(hp-(y+h))*k))
        else:
            self._arc(xc - r*myArc, yc + r, xc - r, yc + r*myArc, xc - r, yc)

        xc = x+r 
        yc = y+r
        self._out('%.2F %.2F l' % (x*k,(hp-yc)*k))
        if '1' not in corners:
            self._out('%.2F %.2F l' % (x*k,(hp-y)*k))
            self._out('%.2F %.2F l' % ((x+r)*k,(hp-y)*k))
        else:
            self._arc(xc - r, yc - r*myArc, xc - r*myArc, yc - r, xc, yc - r)
        self._out(op)
    
    def _arc(self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> None:
        """Draws a Bézier curve arc for rounded corners."""    
        h = self.h
        self._out('%.2F %.2F %.2F %.2F %.2F %.2F c ' % (x1*self.k, (h-y1)*self.k,
            x2*self.k, (h-y2)*self.k, x3*self.k, (h-y3)*self.k))