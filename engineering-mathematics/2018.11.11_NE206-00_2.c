#include <stdio.h>

double f(double x, double y) {	// = x / y
	return x / y;
}

double y(double n, double x) {	// y = nx
	return n * x;
}

int main()
{
	double xl, xr, dx;
	double yl, yr, dy;
	double al, ar;
	double xstar, ystar;
	int n, i, j;
	double area, sum = 0;
	printf("x1, x2를 입력하세요: ");
	scanf("%lf%lf", &xl, &xr);
	printf("y = ax일 때 a1, a2를 입력하세요: ");
	scanf("%lf%lf", &al, &ar);
	printf("n을 입력하세요: ");
	scanf("%d", &n);	// n = 50 or n = 100

	dx = (xr - xl) / n;
	for (i = 0; i < n; i++, xl += dx) {
		xstar = xl + dx / 2;
		yl = y(al, xstar);
		yr = y(ar, xstar);
		dy = (yr - yl) / n;
		area = 0;
		for (j = 0; j < n; j++, yl += dy) {
			ystar = yl + dy / 2;
			area += f(xstar, ystar) * dx * dy;
		}
		sum += area;
	}
	printf("sum: %lf\n", sum);
	return 0;
}