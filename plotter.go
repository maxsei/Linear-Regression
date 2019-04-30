package main

import (
	"log"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// plotRegression takes plotter.XYs pairs for bo
func plotRegression(pts plotter.XYs, linepts plotter.XYs, fname string) {
	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}
	p.X.Label.Text = "TV"
	p.Y.Label.Text = "Sales"
	p.Add(plotter.NewGrid())
	// Add the scatter plot points for the observations.
	s, err := plotter.NewScatter(pts)
	if err != nil {
		log.Fatal(err)
	}
	s.GlyphStyle.Radius = vg.Points(3)
	// Add the line plot points for the predictions.
	l, err := plotter.NewLine(linepts)
	if err != nil {
		log.Fatal(err)
	}
	l.LineStyle.Width = vg.Points(1)
	l.LineStyle.Dashes = []vg.Length{vg.Points(5), vg.Points(5)}
	// Save the plot to a PNG file.
	p.Add(s, l)
	if err := p.Save(4*vg.Inch, 4*vg.Inch, fname); err != nil {
		log.Fatal(err)
	}
}
