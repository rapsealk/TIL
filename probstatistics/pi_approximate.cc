#define _USE_MATH_DEFINES
#include <iostream>
#include <ctime>
#include <cmath>
using namespace std;

double f(double x, double y);

int main()
{
	time_t now;
	time(&now);
	srand((unsigned int) now);
	// srand((unsigned)time(NULL));

	int N, M = 0;
	cout << "반복 횟수 N: ";
	cin >> N;

	for (int i = 0; i < N; ++i) {
		double x = (double)rand() / RAND_MAX;
		double y = (double)rand() / RAND_MAX;

		// (x-1)^2 + (y-1)^2 <= 1
		if (f(x, y) <= 1.0) M += 1;
	}
	cout << "pi: " << M_PI << endl;
	cout << "pi / 4: " << M_PI_4 << endl;
	cout << "M / N: " << (double)M / N << endl;
	// (1 / 4) * pi
	cout << "result: " << (1.0 / 4) * ((double)M / N) << endl;
	return 0;
}

double f(double x, double y) {
	return pow(x-1, 2.0) + pow(y-1, 2.0);
}