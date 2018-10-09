#include <iostream>
#include <string>
using namespace std;

class Matrix {
public:
	Matrix(string name, int values[], int size) {
		this->name = name;
		this->size = size;
		value = new int*[this->size];
		for (int i = 0; i < this->size; i++) value[i] = new int[this->size];
		for (int i = 0; i < this->size; i++) for (int j = 0; j < this->size; j++) value[i][j] = values[i*2 + j];
	}
	~Matrix() {
		for (int i = 0; i < this->size; i++) delete value[i];
		delete value;
	}
	Matrix* operator*(Matrix& other) {
		int *values = new int[size*size];
		memset(values, 0, sizeof(int) * size * size);
		Matrix* mat = new Matrix("C", values, size);
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				// for k = 0 ~ size
				// [i][j] <- [i][k] * [k][j]
				// [0][0] <- [0][0] * [0][0] + [0][1] * [1][0]
				// [0][1] <- [0][0] * [0][1] + [0][1] * [1][1]
				// [1][0] <- [1][0] * [0][0] + [1][1] * [1][0]
				// [1][1] <- [1][0] * [0][1] + [1][1] * [1][1]
				int element = 0;
				for (int k = 0; k < size; k++) element += value[i][k] * other.value[k][j];
				mat->value[i][j] = element;
			}
		}
		delete values;
		return mat;
	}
	void print() {
		cout << "Matrix [" << this->name << "] =====" << endl;
		for (int i = 0; i < size; i++) {
			cout << "[ ";
			for (int j = 0; j < size; j++) {
				cout << value[i][j] << ' ';
			}
			cout << " ]" << endl;
		}
		cout << endl;
	}
private:
	string name;
	int size;
	int **value = nullptr;
};

int main()
{
	int A[] = { 4, 7, 3, 5 };
	int B[] = { 9, -2, 6, 8 };
	int size = sizeof(A) / sizeof(int) / 2;
	Matrix a = Matrix("A", A, size);
	Matrix b = Matrix("B", B, size);
	Matrix* c = a * b;
	a.print();
	b.print();
	c->print();
	return 0;
}