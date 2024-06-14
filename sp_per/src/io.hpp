#pragma once

bool read_matrix(double *&matrix, int &n, int &nonzeros, const char *fname);
void print_matrix(double *matrix, int size);