package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"strconv"

	plot "gonum.org/v1/plot"
	plotter "gonum.org/v1/plot/plotter"
	//  reflect
	vg "gonum.org/v1/plot/vg"
)

//maybe we need to switch to the leveng
func main() {
	showStep := flag.Bool("step", false, "prints out each iteration of the regression")
	data := flag.String("data", "regression_test.csv", "path to csv data source where only the first two columns are read")
	outputLine := flag.String("output", "value", "the name of the file that the line is outputted to. Must be *.png,*jpeg. *bmp")
	flag.Parse()

	f, err := os.Open(*data)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	reader := csv.NewReader(f) //*csv.Reader

	//fmt.Println(reflect.TypeOf(reader))
	X, Y, head := readInValues(reader)
	fmt.Println(head)
	for i := range X {
		fmt.Printf("%.3f, %.3f\n", X[i], Y[i])
	}

	fmt.Println("Start Regression:")
	m, b := newtonsRegression(X, Y, 0.00000001, *showStep)
	r := correlationCoefficient(X, Y)
	fmt.Printf("\ny = %.8fx + %.8f\tCorrelation Coefficient: %.8f\tMAE: %.8f\n", m, b, r, calcMAE(X, Y, m, b))

	pts := make(plotter.XYs, len(X))
	ptsPred := make(plotter.XYs, len(X))
	for i := range X {
		pts[i].X = X[i]
		pts[i].Y = Y[i]
		ptsPred[i].X = X[i]
		ptsPred[i].Y = predict(m, X[i], b)
	}
	plotRegression(pts, ptsPred, *outputLine)

}
func readInValues(reader *csv.Reader) (X, Y []float64, head string) {
	line := 1
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if line == 1 {
			line++
			for _, v := range record {
				head += v + " "
			}
			continue
		}
		XVal, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			log.Printf("Parseing line %d failed, enexpected type\n", line)
			continue
		}

		YVal, err := strconv.ParseFloat(record[1], 64)
		if err != nil {
			log.Printf("Parseing line %d failed, enexpected type\n", line)
			continue
		}
		X = append(X, XVal)
		Y = append(Y, YVal)
		line++
	}
	return
}
func correlationCoefficient(X, Y []float64) float64 {
	var sumSqX, sumSqY, sumXY, sumX, sumY float64
	for i := range X {
		sumSqX += math.Pow(X[i], 2)
		sumSqY += math.Pow(Y[i], 2)
		sumXY += X[i] * Y[i]
		sumX += X[i]
		sumY += Y[i]
	}
	n := float64(len(X))
	return (n*sumXY - sumX*sumY) /
		math.Pow((n*sumSqX-math.Pow(sumX, 2))*(n*sumSqY-math.Pow(sumY, 2)), .5)
}
func plotRegression(pts plotter.XYs, ptsPred plotter.XYs, fname string) {
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
	l, err := plotter.NewLine(ptsPred)
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
func predict(m, x, b float64) float64 {
	return m*x + b
}
func calcMAE(X, Y []float64, m, b float64) (mAE float64) {
	for i := range Y {
		mAE += math.Abs(Y[i]-predict(m, X[i], b)) / float64(len(Y))
	}
	return
}
func newtonsRegression(xx, yy []float64, epsilon float64, show bool) (float64, float64) {
	// define m and b as well as their changes, the magnitude of those changes,
	// and the number of iterations counted
	var m, b, deltaM, deltaB, heshMagnitude, iterations float64
	// loop until magnitude is lower than epsilon except for the first iteration
	for heshMagnitude > epsilon || iterations == 0 {
		// calculate changes in m and b as well as calculating their combined magnitude
		deltaM, deltaB = stepNewton(xx, yy, m, b)
		heshMagnitude = math.Pow((math.Pow(deltaM, 2) + math.Pow(deltaB, 2)), .5)
		if show {
			fmt.Printf("Iteration: %.0f\t%cm: %.8f\t%cb: %.8f\t |%cf|: %.8f\tm: %.16f\tb: %.16f\n",
				iterations, 0x0394, deltaM, 0x0394, deltaB, 0x0394, heshMagnitude, m, b)
		}
		//why are these signs different. Wolfe Conditions?
		m = m + deltaM
		b = b - deltaB
		iterations++
	}
	return m, b
}
func stepNewton(X, Y []float64, m, b float64) (float64, float64) {
	// dfine all partial derivates
	var fm, fb, fmm, fbb, fmb float64
	// sum up change in error with respect to m, b, m^2, and m*b
	// note that b^2 is not here because it is constant at 2.0
	for i := range X {
		fm += -X[i] * (Y[i] - (m*X[i] + b))
		fb += -(Y[i] - (m*X[i] + b))
		fmm += X[i] * X[i]
		fmb += X[i]
	}
	// multiply all partials by the two constant that is almost always
	// pulled out of the derivative equation and devide by n to normalize
	// note that the b^2 derivates is always a constant values of 2
	n := float64(len(X))
	fm = 2 * fm / n
	fb = 2 * fb / n
	fmm = fmm / n
	fbb = 2.0
	fmb = 2 * fmb / n
	// calculate the determinant of the hessian matrix and the change of m and b
	determ := fmm*fbb - math.Pow(fmb, 2)
	dm := (fm*fbb + fb*fmb) / determ
	db := (fm*fmb + fb*fmm) / determ
	return dm, db
}
