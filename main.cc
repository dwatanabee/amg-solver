#include <iostream>
#include <string>
#include <numeric>
#include <fstream>
#include "smoother.h"
#include "amg_solver.h"

using namespace std;
using namespace Eigen;

MatrixXd read_matrix(const char *file)
{
  int m, n, r, c;
  double v;
  char lb, rb, co;
  std::ifstream is(file);
  is >> m >> n;
  MatrixXd A = MatrixXd::Zero(m, n);
  while (is >> lb >> r >> co >> c >> rb >> v)
  {
    A(r - 1, c - 1) = v;
  }
  return A;
}

int test_smoother()
{
  srand(time(NULL));

  MatrixXd A = MatrixXd::Random(10000, 10000);
  for (size_t i = 0; i < A.rows(); ++i)
    A(i, i) += 5.0;
  const size_t sp_ratio = 0.001;
  const size_t zero_count = A.rows() * A.cols() * (1.0 - sp_ratio);
  for (size_t cnt = 0; cnt < 5 * zero_count; ++cnt)
  {
    size_t I = rand() % 10000;
    size_t J = rand() % 10000;
    if (I != J)
      A(I, J) = 0;
  }

  VectorXd rhs = VectorXd::Random(A.cols());
  std::cout << rhs.transpose().head(20) << endl
            << endl;

  SparseMatrix<double, RowMajor> Ar = A.sparseView();
  VectorXd y = VectorXd::Random(A.cols());
#define GAUSS_SEIDEL 0
#if GAUSS_SEIDEL
  shared_ptr<amg::smoother> smooth(new amg::gauss_seidel);
#else
  shared_ptr<amg::smoother> smooth(new amg::damped_jacobi);
#endif
  for (size_t i = 0; i < 10000; ++i)
    smooth->apply_prev_smooth(Ar, rhs, y, nullptr);
  std::cout << (Ar * y).transpose().head(20) << endl
            << endl;

  std::cout << "done\n";
  return 0;
}

int test_red_black_gs()
{
  srand(time(NULL));

  MatrixXd A = MatrixXd::Random(10000, 10000);
  for (size_t i = 0; i < A.rows(); ++i)
    A(i, i) += 5.0;
  const size_t sp_ratio = 0.001;
  const size_t zero_count = A.rows() * A.cols() * (1.0 - sp_ratio);
  for (size_t cnt = 0; cnt < 5 * zero_count; ++cnt)
  {
    size_t I = rand() % 10000;
    size_t J = rand() % 10000;
    if (I != J)
      A(I, J) = 0;
  }

  VectorXd rhs = VectorXd::Random(A.cols());
  std::cout << rhs.transpose().head(20) << endl
            << endl;

  SparseMatrix<double, RowMajor> Ar = A.sparseView();
  VectorXd y = VectorXd::Random(A.cols());
  vector<bool> tag;
  amg::amg_solver::tag_red_black(Ar, tag);
  /// see red-black tag
  for (size_t i = 0; i < tag.size(); ++i)
    std::cout << tag[i] << " ";
  std::cout << endl
            << endl;
  shared_ptr<amg::smoother> smooth(new amg::red_black_gauss_seidel);
  for (size_t cnt = 0; cnt < 10000; ++cnt)
    smooth->apply_prev_smooth(Ar, rhs, y, &tag);
  std::cout << (Ar * y).transpose().head(20) << endl
            << endl;

  std::cout << "done\n";
  return 0;
}

int test_amg_solver()
{
  srand(125482);

  cout << "# info: construct system matrix\n";
  const size_t size = 10455;
  MatrixXd A = MatrixXd::Random(size, size);
  for (size_t i = 0; i < A.rows(); ++i)
    A(i, i) += 5.0;
  const size_t sp_ratio = 0.001;
  const size_t zero_count = A.rows() * A.cols() * (1.0 - sp_ratio);
  for (size_t cnt = 0; cnt < 5 * zero_count; ++cnt)
  {
    size_t I = rand() % size;
    size_t J = rand() % size;
    if (I != J)
      A(I, J) = 0;
  }

  VectorXd rhs = VectorXd::Random(A.cols());
  cout << rhs.transpose().segment<20>(1000) << endl
       << endl;

  SparseMatrix<double, RowMajor> Ar = A.sparseView();
  VectorXd x;
  shared_ptr<amg::amg_solver> sol = std::make_shared<amg::amg_solver>();
  cout << "# info: AMG compute\n";
  sol->compute(Ar);
  cout << "# info: AMG solve\n";

  // if (pt.get<string>("multigrid.value") == "FMG")
  sol->solveFMG(rhs, x);
  // else if (pt.get<string>("multigrid.value") == "VC")
  //   sol->solve(rhs, x);

  cout << (Ar * x).transpose().segment<20>(1000) << endl
       << endl;

  cout << "done\n";
  return 0;
}

int test_std_algorithm()
{
  Matrix<char, 10, 1> v;
  std::fill(v.data(), v.data() + v.size(), 'A');
  cout << v.transpose() << endl;
  std::replace(v.data(), v.data() + v.size(), 'A', 'B');
  cout << v.transpose() << endl;

  vector<size_t> ones{213, 23, 1, 0, 0};
  std::partial_sum(ones.begin(), ones.end(), ones.begin());
  for (auto it : ones)
    cout << it << " ";
  cout << endl;

  cout << "done\n";
  return 0;
}

int main(int argc, char *argv[])
{

  test_smoother();
  // test_red_black_gs();
  // test_amg_solver();
  // test_std_algorithm();
  //    CALL_SUB_PROG(test_amgcl);

  return 0;
}
