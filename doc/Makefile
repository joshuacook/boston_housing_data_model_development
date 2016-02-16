pdf:
	pandoc boston_housing.md --latex-engine=xelatex -o boston_housing.pdf

watch:
	fswatch boston_housing.md | xargs -n 1 make pdf
