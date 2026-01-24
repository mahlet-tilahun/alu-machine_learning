#!/usr/bin/env python3
""" a function def matrix_transpose(matrix): that returns the transpose of a 2D matrix, matrix:"""

def matrix_transpose(matrix):
    
    matrix_transpose = []
    for index, row in enumerate(matrix):
        if index is 0:
            for i in row:
                matrix_transpose.append([])
        for idx, i in enumerate(row):
            matrix_transpose[idx].append(i)
    return matrix_transpose
