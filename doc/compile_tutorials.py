import os
import shutil

from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
from nbconvert.exporters import RSTExporter
from nbconvert.writers import FilesWriter
import nbformat

from PIL import Image, ImageChops

def compile_tutorial(tutorial_name, force_recompile=False):
	print('- Compiling tutorial ' + tutorial_name + '...')

	notebook_path = 'tutorial_notebooks/' + tutorial_name + '/' + tutorial_name + '.ipynb'
	export_path = 'tutorials/' + tutorial_name + '/' + tutorial_name
	thumb_dest = os.path.dirname(export_path) + '/thumb.png'

	if not os.path.exists(os.path.dirname(export_path)):
		os.makedirs(os.path.dirname(export_path))
	
	# Read in notebook
	notebook = nbformat.read(notebook_path, 4)

	# Scrape title, description and thumbnail
	first_cell = notebook.cells[0]

	title = first_cell.source.splitlines()[0]
	if '#' in title:
		title = title.replace('#', '').strip()
	
	description = first_cell.source.splitlines()[2].strip()
	
	if 'thumbnail_figure_index' in notebook.metadata:
		thumbnail_figure_index = notebook.metadata['thumbnail_figure_index']
	else:
		thumbnail_figure_index = -1
	
	if 'level' in notebook.metadata:
		level = notebook.metadata['level'].capitalize()
	elif 'difficulty' in notebook.metadata:
		level = notebook.metadata['difficulty'].capitalize()
	else:
		level = 'Unknown'
	
	# Check if the tutorial was already compiled.
	if os.path.exists(export_path + '.rst'):
		if os.path.getmtime(export_path + '.rst') > os.path.getmtime(notebook_path):
			if force_recompile:
				print('  Already compiled. Recompiling anyway...')
			else:
				print('  Already compiled. Skipping...')
				return title, level, description, thumb_dest.split('/', 1)[-1]

	# Execute notebook if not already executed
	already_executed = any(c.get('outputs') or c.get('execution_count') for c in notebook.cells if c.cell_type == 'code')

	resources = {}
	
	if not already_executed:
		ep = ExecutePreprocessor(timeout=120, kernel_name='python3')
		try:
			notebook, resources = ep.preprocess(notebook, resources={'metadata': {'path': os.path.dirname(notebook_path)}})
		except CellExecutionError as err:
			print('Error while processing notebook.')
			print(err)

	exporter = RSTExporter()
	output, resources = exporter.from_notebook_node(notebook, resources)

	writer = FilesWriter(build_directory=os.path.dirname(export_path))
	writer.write(output, resources, notebook_name=os.path.basename(export_path))

	pictures = sorted(resources['outputs'], key=output.find)

	try:
		thumbnail_source = pictures[thumbnail_figure_index]

		# Read in thumbnail source image
		img = Image.open(os.path.dirname(export_path) + '/' + thumbnail_source)

		# Trim whitespace
		bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
		diff = ImageChops.difference(img, bg)
		diff = ImageChops.add(diff, diff)
		bbox = diff.getbbox()
		if bbox:
			img = img.crop(bbox)

		# Resize image to have a width of 400px
		img.thumbnail([400, 1000])

		# Save thumbnail
		img.save(thumb_dest)
	except:
		shutil.copyfile('_static/no_thumb.png', thumb_dest)
	
	print('  Done!')

	return title, level, description, thumb_dest.split('/', 1)[-1]

index_preamble = '''
Tutorials
=========

These tutorials demonstrate the features of HCIPy in the context of a standard workflow.
'''

entry_template = '''
.. only:: html

	.. container:: tutorial_item

		:doc:`{title} <{name}/{name}>`

		.. container:: tutorial_row

			.. container:: tutorial_thumbnail

				.. figure:: {thumbnail_file}

			.. container:: tutorial_description

				**Level:** {level}
				
				**Description:** {description}
'''

def compile_all_tutorials():
	print('Compling all tutorials...')

	tutorials = {}
	tutorial_names = sorted(os.listdir('tutorial_notebooks/'))
	tutorial_names = [name for name in tutorial_names if 'checkpoint' not in name]

	for name in tutorial_names:
		tutorials[name] = compile_tutorial(name)

	# Sort by level
	levels = ['Beginner', 'Intermediate', 'Advanced', 'Expert', 'Unknown']
	tutorial_names = sorted(tutorial_names, key=lambda name: levels.index(tutorials[name][1]))

	f = open('tutorials/index.rst', 'w')
	f.write(index_preamble)

	# Write toctree
	f.write('\n.. toctree::\n    :maxdepth: 1\n    :hidden:\n\n')
	for name in tutorial_names:
		f.write('    ' + name + '/' + name + '\n')
	f.write('\n\n')
	
	# Write list
	for name in tutorial_names:
		title, level, desc, thumb = tutorials[name]
		f.write(entry_template.format(thumbnail_file=thumb, title=title, level=level, description=desc, name=name))
	
	f.close()
