#include <stdio.h>

// 5.2.4
// y` = f(x, y) = 2xy
// y(x_0) = y_0
// y(1) = 1
// x_n = x_0 + nh
// y_n+1 = y_n + hf(x_n, y_n)

// f(x, y)는 별도의 분리된 함수로 만들어서 나중에 쉽게 바꿀 수 있도록 작성할 것.
double f(double x, double y) {	// = y` = f(x, y) = 2xy
	return 2 * x * y;
}

int main()
{
	double x0, y0, h;
	double xn, yn = 0;
	double x;
	double k1, k2;
	printf("x0, y0: ");
	scanf("%lf%lf", &x0, &y0);
	printf("등간격 h: ");
	scanf("%lf", &h);
	printf("원하는 해의 위치 x: ");
	scanf("%lf", &x);

	// Euler
	for (xn = x0, yn = y0; xn <= x; xn += h) {
		yn += h * f(xn, yn);
	}
	printf("[Euler]\n");
	printf("xn: %.3lf, yn: %.3lf\n", xn, yn);

	// Heun
	for (xn = x0, yn = y0; xn <= x; xn += h) {
		k1 = h * f(xn, yn);
		k2 = h * f(xn + ((double)2 / 3) * h, yn + ((double)2 / 3) * k1);
		yn += (k1 + 3 * k2) / 4;
	}
	printf("[Heun]\n");
	printf("xn: %.3lf, yn: %.3lf\n", xn, yn);

	return 0;
}