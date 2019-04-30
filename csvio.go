package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"strconv"
	"strings"
)

// parseColumns parseColumns will get the columns of interest from the columns
// string,  will return an error the column string is invalid
func parseColumns(colstr string) (int, int, error) {
	var xcol, ycol int
	colErr := fmt.Errorf("invalid columns string format: %s must be in form %s where columns are different values", colstr, "0,1")
	colSlice := strings.Split(colstr, ",")
	if len(colSlice) != 2 {
		fmt.Println("!=2")
		return -1, -1, colErr
	}
	if i, err := strconv.Atoi(colSlice[0]); err == nil {
		xcol = i
	}
	if i, err := strconv.Atoi(colSlice[1]); err == nil {
		ycol = i
	}
	if xcol == ycol {
		return -1, -1, colErr
	}
	return xcol, ycol, nil
}

// describeString will return a string that has the up to the first five entries
// of the values of interest along with header information for both columns
func describeString(X, Y []float64, head string) string {
	result := head + "\n"
	for i := 0; i < 5; i++ {
		if i >= len(X) {
			continue
		}
		result += fmt.Sprintf("%.3f\t%.3f\n", X[i], Y[i])
	}
	return result
}

// readInValues takes in a csv reader and returns a two vectors for x and y
// values respectively. The function will also skip over rows that have missing
// data for row column values that are of interest. Also returns header
// information of the csv file if their is any (headers cannot contain labels
// that can be parsed intto numeric types)
func readInValues(xcol, ycol int, reader *csv.Reader) ([]float64, []float64, string) {
	var X []float64
	var Y []float64
	var head string
	line := 1
	for {
		// read lines of csv until reaching the end
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		// handle header labels
		if line == 1 {
			line++
			if _, err := strconv.Atoi(record[xcol]); err == nil {
				head = ""
				continue
			}
			head += record[xcol] + "\t"
			if _, err := strconv.Atoi(record[ycol]); err == nil {
				head = ""
				continue
			}
			head += record[xcol]
			continue
		}
		// if there is a missing value in any row column that is being parsed
		// the loop will skip the row entirely
		XVal, err := strconv.ParseFloat(record[xcol], 64)
		if err != nil {
			log.Printf("Parseing line %d failed, enexpected type\n", line)
			continue
		}
		YVal, err := strconv.ParseFloat(record[ycol], 64)
		if err != nil {
			log.Printf("Parseing line %d failed, enexpected type\n", line)
			continue
		}
		X = append(X, XVal)
		Y = append(Y, YVal)
		line++
	}
	return X, Y, head
}
