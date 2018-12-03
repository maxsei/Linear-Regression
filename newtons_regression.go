package main

import(
    "encoding/csv"
    "fmt"
    "log"
    "math"
    "io"
    "os"
    "strconv"
    plot "gonum.org/v1/plot"
    plotter"gonum.org/v1/plot/plotter"
//  reflect
    vg "gonum.org/v1/plot/vg"
)
//maybe we need to switch to the leveng
func main(){
    fmt.Println("Start Regression:")

    resPath := `C:\Users\owner\Documents\Max\AtomWorkspace\MachineLearningInGolang\Resources\`
    f, err := os.Open(resPath+"regression_test.csv")
    if err != nil {
    log.Fatal(err)
    }
    defer f.Close()
    reader := csv.NewReader(f)//*csv.Reader

    //fmt.Println(reflect.TypeOf(reader))
    X,Y,head :=readInValues(reader)
    fmt.Println(head)
    for i := range X{
        fmt.Printf("%.3f, %.3f\n",X[i], Y[i])
    }

    m,b := newtonsRegression(X,Y,0.00000001)
    r := correlationCoefficient(X,Y)
    fmt.Printf("\ny = %.8fx + %.8f\tCorrelation Coefficient: %.8f\tMAE: %.8f\n",m, b, r,calcMAE(X, Y, m, b))

    pts := make(plotter.XYs, len(X))
    ptsPred := make(plotter.XYs, len(X))
    for i := range X{
        pts[i].X = X[i]
        pts[i].Y = Y[i]
        ptsPred[i].X = X[i]
        ptsPred[i].Y = predict(m,X[i],b)
    }
    plotRegression(pts, ptsPred)

}
func readInValues(reader *csv.Reader) (X,Y[]float64, head string){
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
func correlationCoefficient(X, Y[]float64) float64{
    var sumSqX, sumSqY, sumXY, sumX, sumY float64
    for i := range X{
        sumSqX += math.Pow(X[i], 2)
        sumSqY += math.Pow(Y[i], 2)
        sumXY += X[i]*Y[i]
        sumX += X[i]
        sumY += Y[i]
    }
    n :=float64(len(X))
    return (n*sumXY-sumX*sumY)/
    math.Pow( (n*sumSqX-math.Pow(sumX,2)) * (n*sumSqY-math.Pow(sumY,2)) , .5)
}
func plotRegression(pts plotter.XYs, ptsPred plotter.XYs){
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
    thisPath:=`C:\Users\owner\Documents\Max\AtomWorkspace\MachineLearningInGolang\Regression\hand_made_regressor/`
    if err := p.Save(4*vg.Inch, 4*vg.Inch, thisPath+"regression_line.png"); err != nil {
        log.Fatal(err)
    }
}
func predict(m, x, b float64) float64{
    return m*x+b
}
func calcMAE(X, Y []float64,m, b float64) (mAE float64){
    for i := range Y{
        mAE += math.Abs(Y[i]-predict(m,X[i],b)) / float64(len(Y))
    }
    return
}
func newtonsRegression(X,Y[]float64, epsilon float64)  (m, b float64){//m,b start at 0,0
    itererations := 0
    deltaM,deltaB := newton(X,Y,m,b)
    heshMagnitude := math.Pow((math.Pow(deltaM,2)+math.Pow(deltaB,2)),.5)
    for heshMagnitude > epsilon{
        deltaM,deltaB = newton(X,Y,m,b)
        /*
        if(heshMagnitude < math.Pow((math.Pow(deltaM,2)+math.Pow(deltaB,2)),.5)){
            fmt.Println("Diverging from solution breaking loop")
            break
        }
        */
        heshMagnitude = math.Pow((math.Pow(deltaM,2)+math.Pow(deltaB,2)),.5)
        fmt.Printf("Iteration: %d\t%cm: %.8f\t%cb: %.8f\t |%cf|: %.8f\tm: %.16f\tb: %.16f\n",
        itererations,0x0394, deltaM,0x0394,deltaB,0x0394,heshMagnitude,m,b);

        m = m - deltaM
        b = b + deltaB //why are these signs different. Wolfe Conditions?
        itererations++
    }
    return
}
func newton(X,Y[]float64,m,b float64) (dm, db float64){
    fm, fb, fmm, fbb, fmb, determ := delta(X, Y, m, b)
    //dm = (fm*fbb-fb*fmb)/determ
    //db = (-fm*fmb+fb*fmm)/determ
    dm = (fm*fbb+fb*fmb)/determ
    db = (fm*fmb+fb*fmm)/determ
    return
}
func delta(X,Y[]float64,m,b float64) (fm, fb, fmm, fbb, fmb, determ float64){
    n := float64(len(X))
    for i := range X {
        fm += -X[i]*(Y[i]-(m*X[i]+b))
        fb += -(Y[i]-(m*X[i]+b))
        fmm += X[i]*X[i]
        fmb += X[i]
    }
    fm = 2*fm/n
    fb = 2*fb/n
    fmm = fmm/n
    fbb = 2.0
    fmb = 2*fmb/n
    determ = fmm * fbb - math.Pow(fmb,2)
    return
}
