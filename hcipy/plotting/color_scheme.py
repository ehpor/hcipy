from cycler import cycler
from . import colors

# Blue, red, green, orange, purple, brown
palettes = {
	'dark': cycler(color=[colors.blue_700, colors.red_700, colors.green_700, colors.orange_700, colors.purple_700, colors.brown_700])
}

def set_color_scheme(dark=False, publication_quality=False, cmap='viridis'):
	"""
	Apply a color scheme to all matplotlib figures. The setting
	publication_quality uses LaTeX for all text in the figure. 
	"""
	import matplotlib as mpl

	mpl.rc('lines', linewidth=1.5, markeredgewidth=0.25)
	mpl.rc('image', cmap=cmap)
	mpl.rc('legend', scatterpoints=1, numpoints=1, labelspacing=0.3)
	mpl.rc('axes.formatter', limits=(-4,4))
	mpl.rc('text.latex', preamble=['\\usepackage{amsmath}'])

	mpl.rc('xtick', labelsize='small')
	mpl.rc('ytick', labelsize='small')
	mpl.rc('axes', titlesize='medium', labelsize='medium')
	mpl.rc('legend', fontsize='medium')
	
	mpl.rc('savefig', transparent=True)

	if dark:
		mpl.rc('axes', prop_cycle=palettes['light'], facecolor='k', labelcolor='w', edgecolor='w')
		mpl.rc('xtick', color='w')
		mpl.rc('ytick', color='w')
		mpl.rc('grid', color='w')
		mpl.rc('figure', facecolor='k', edgecolor='k')
		mpl.rc('text', color='w')
	else:
		mpl.rc('axes', prop_cycle=palettes['dark'], facecolor='w', labelcolor='k', edgecolor='k')
		mpl.rc('xtick', color='k')
		mpl.rc('ytick', color='k')
		mpl.rc('grid', color='k')
		mpl.rc('figure', facecolor='w', edgecolor='w')
		mpl.rc('text', color='k')

	if publication_quality:
		mpl.rc('text', usetex=True)
		mpl.rc('font', family='sans-serif')
		mpl.rc('font', serif=['computer modern roman'], monospace=['computer modern typewriter'])
		mpl.rcParams['font.sans-serif'] = ['computer modern sans serif']

		mpl.rc('font', size=11)
		mpl.rc('figure', figsize=(7.2, 5.1))
	else:
		mpl.rc('text', usetex=False)
		mpl.rc('font', family='sans-serif')
		mpl.rc('font', serif=['Bitstream Vera Serif', 'New Century Schoolbook', 'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Times New Roman', 'Times', 'Palatino', 'Charter', 'serif'])
		mpl.rcParams['font.sans-serif'] = ['Bitstream Vera Sans', 'Lucida Grande', 'Verdana', 'Geneva', 'Lucid', 'Arial', 'Helvetica', 'Avant Garde', 'sans-serif']
		mpl.rc('font', monospace=['Bitstream Vera Sans Mono', 'Andale Mono', 'Nimbus Mono L', 'Courier New', 'Courier', 'Fixed', 'Terminal', 'monospace'])

		mpl.rc('figure', figsize=(10, 7.1))
		mpl.rc('font', size=14)

set_color_scheme(False, False, 'viridis')