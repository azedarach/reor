#ifndef REOR_EIGEN_BACKEND_HPP_INCLUDED
#define REOR_EIGEN_BACKEND_HPP_INCLUDED

/**
 * @file eigen_backend.hpp
 * @brief provides implementation of Eigen3 backend
 */

#include "reor/backend_interface.hpp"
#include "reor/backends/eigen_type_traits.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <type_traits>

namespace reor {

namespace backends {

template <class Scalar>
struct Eigen_backend {
   using Index = Eigen::Index;
   using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

   static Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
   create_matrix(Index rows, Index cols)
      {
         return Eigen::Matrix<
            Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(rows, cols);
      }

   static Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
   copy_matrix(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& M)
      {
         return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(M);
      }

   static Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
   create_constant_matrix(Index rows, Index cols, Scalar value)
      {
         return Eigen::Matrix<
            Scalar, Eigen::Dynamic, Eigen::Dynamic>::Constant(
               rows, cols, value);
      }

   static Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
   create_diagonal_matrix(Index size, Scalar diagonal = Scalar(1))
      {
         return diagonal * Eigen::Matrix<
            Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(size, size);
      }
};

namespace detail {

template <class Matrix>
struct matrix_traits_impl<
   Matrix,
   typename std::enable_if<is_eigen_matrix<Matrix>::value>::type> {

   typedef typename Eigen::Index index_type;
   typedef typename Matrix::Scalar element_type;
   typedef typename Matrix::RealScalar real_element_type;

};

template <class Matrix>
struct rows_impl<
   Matrix,
   typename std::enable_if<is_eigen_matrix<Matrix>::value>::type> {

   static std::size_t get(const Matrix& m)
      {
         return static_cast<std::size_t>(m.rows());
      }
};

template <class Matrix>
struct cols_impl<
   Matrix,
   typename std::enable_if<is_eigen_matrix<Matrix>::value>::type> {

   static std::size_t get(const Matrix& m)
      {
         return static_cast<std::size_t>(m.cols());
      }
};

template <class Index, class Scalar, class Matrix>
struct set_matrix_element_impl<
   Index, Scalar, Matrix,
   typename std::enable_if<is_eigen_matrix<Matrix>::value>::type> {

   static void set(Index row, Index col, Scalar s, Matrix& A)
      {
         A(row, col) = s;
      }
};

template <class Index, class Matrix>
struct get_matrix_element_impl<
   Index, Matrix,
   typename std::enable_if<is_eigen_matrix<Matrix>::value>::type> {

   using return_type = typename Matrix::Scalar;

   static return_type get(Index row, Index col, const Matrix& A)
      {
         return A(row, col);
      }
};

template <class Scalar, class Matrix>
struct add_constant_impl<
   Scalar, Matrix,
   typename std::enable_if<is_eigen_matrix<Matrix>::value>::type> {

   static void eval(Scalar c, Matrix& A)
      {
         A.colwise() += Eigen::VectorXd::Constant(A.rows(), c);
      }
};

template <class Scalar1, class MatrixA, class Scalar2, class MatrixB,
          class MatrixC>
struct geam_impl<
   Scalar1, MatrixA, Scalar2, MatrixB, MatrixC,
   typename std::enable_if<is_eigen_matrix<MatrixA>::value &&
                           is_eigen_matrix<MatrixB>::value &&
                           is_eigen_matrix<MatrixC>::value>::type> {

   static void eval(Scalar1 alpha, const MatrixA& A, Scalar2 beta,
                    const MatrixB& B, MatrixC& C,
                    Op_flag opA, Op_flag opB)
      {
         if (opA == Op_flag::None && opB == Op_flag::None) {
            C = alpha * A + beta * B;
         } else if (opA == Op_flag::Transpose && opB == Op_flag::None) {
            C = alpha * A.transpose() + beta * B;
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::None) {
            C = alpha * A.adjoint() + beta * B;
         } else if (opA == Op_flag::None && opB == Op_flag::Transpose) {
            C = alpha * A + beta * B.transpose();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Transpose) {
            C = alpha * A.transpose() + beta * B.transpose();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Transpose) {
            C = alpha * A.adjoint() + beta * B.transpose();
         } else if (opA == Op_flag::None && opB == Op_flag::Adjoint) {
            C = alpha * A + beta * B.adjoint();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Adjoint) {
            C = alpha * A.transpose() + beta * B.adjoint();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Adjoint) {
            C = alpha * A.adjoint() + beta * B.adjoint();
         }
      }
};

template <class Scalar1, class MatrixA, class MatrixB, class Scalar2,
          class MatrixC>
struct gemm_impl<
   Scalar1, MatrixA, MatrixB, Scalar2, MatrixC,
   typename std::enable_if<is_eigen_matrix<MatrixA>::value &&
                           is_eigen_matrix<MatrixB>::value &&
                           is_eigen_matrix<MatrixC>::value>::type> {

   static void eval(Scalar1 alpha, const MatrixA& A, const MatrixB& B,
                    Scalar2 beta, MatrixC& C, Op_flag opA, Op_flag opB)
      {
         if (opA == Op_flag::None && opB == Op_flag::None) {
            C = alpha * A * B + beta * C;
         } else if (opA == Op_flag::Transpose && opB == Op_flag::None) {
            C = alpha * A.transpose() * B + beta * C;
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::None) {
            C = alpha * A.adjoint() * B + beta * C;
         } else if (opA == Op_flag::None && opB == Op_flag::Transpose) {
            C = alpha * A * B.transpose() + beta * C;
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Transpose) {
            C = alpha * A.transpose() * B.transpose() + beta * C;
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Transpose) {
            C = alpha * A.adjoint() * B.transpose() + beta * C;
         } else if (opA == Op_flag::None && opB == Op_flag::Adjoint) {
            C = alpha * A * B.adjoint() + beta * C;
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Adjoint) {
            C = alpha * A.transpose() * B.adjoint() + beta * C;
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Adjoint) {
            C = alpha * A.adjoint() * B.adjoint() + beta * C;
         }
      }
};

template <class Scalar1, class MatrixA, class MatrixB, class Scalar2,
          class MatrixC>
struct trace_gemm_impl<
   Scalar1, MatrixA, MatrixB, Scalar2, MatrixC,
   typename std::enable_if<is_eigen_matrix<MatrixA>::value &&
                           is_eigen_matrix<MatrixB>::value &&
                           is_eigen_matrix<MatrixC>::value>::type> {

   using value_type = typename std::common_type<
      Scalar1, typename MatrixA::Scalar, typename MatrixB::Scalar,
      Scalar2, typename MatrixC::Scalar>::type;

   static value_type eval(Scalar1 alpha, const MatrixA& A, const MatrixB& B,
                          Scalar2 beta, MatrixC& C, Op_flag opA, Op_flag opB)
      {
         if (opA == Op_flag::None && opB == Op_flag::None) {
            return (alpha * A * B + beta * C).trace();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::None) {
            return (alpha * A.transpose() * B + beta * C).trace();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::None) {
            return (alpha * A.adjoint() * B + beta * C).trace();
         } else if (opA == Op_flag::None && opB == Op_flag::Transpose) {
            return (alpha * A * B.transpose() + beta * C).trace();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Transpose) {
            return (alpha * A.transpose() * B.transpose() + beta * C).trace();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Transpose) {
            return (alpha * A.adjoint() * B.transpose() + beta * C).trace();
         } else if (opA == Op_flag::None && opB == Op_flag::Adjoint) {
            return (alpha * A * B.adjoint() + beta * C).trace();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Adjoint) {
            return (alpha * A.transpose() * B.adjoint() + beta * C).trace();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Adjoint) {
            return (alpha * A.adjoint() * B.adjoint() + beta * C).trace();
         }
      }
};

template <class Scalar1, class MatrixA, class MatrixB>
struct trace_gemm_op_impl<
   Scalar1, MatrixA, MatrixB,
   typename std::enable_if<is_eigen_matrix<MatrixA>::value &&
                           is_eigen_matrix<MatrixB>::value>::type> {

   using value_type = typename std::common_type<
      Scalar1, typename MatrixA::Scalar, typename MatrixB::Scalar>::type;

   static value_type eval(Scalar1 alpha, const MatrixA& A, const MatrixB& B,
                          Op_flag opA, Op_flag opB)
      {
         if (opA == Op_flag::None && opB == Op_flag::None) {
            return (alpha * A * B).trace();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::None) {
            return (alpha * A.transpose() * B).trace();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::None) {
            return (alpha * A.adjoint() * B).trace();
         } else if (opA == Op_flag::None && opB == Op_flag::Transpose) {
            return (alpha * A * B.transpose()).trace();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Transpose) {
            return (alpha * A.transpose() * B.transpose()).trace();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Transpose) {
            return (alpha * A.adjoint() * B.transpose()).trace();
         } else if (opA == Op_flag::None && opB == Op_flag::Adjoint) {
            return (alpha * A * B.adjoint()).trace();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Adjoint) {
            return (alpha * A.transpose() * B.adjoint()).trace();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Adjoint) {
            return (alpha * A.adjoint() * B.adjoint()).trace();
         }

         throw std::runtime_error("invalid combination of operations");
      }
};

template <class Scalar1, class MatrixA, class MatrixB, class Scalar2,
          class MatrixC>
struct sum_gemm_impl<
   Scalar1, MatrixA, MatrixB, Scalar2, MatrixC,
   typename std::enable_if<is_eigen_matrix<MatrixA>::value &&
                           is_eigen_matrix<MatrixB>::value &&
                           is_eigen_matrix<MatrixC>::value>::type> {

   using value_type = typename std::common_type<
      Scalar1, typename MatrixA::Scalar, typename MatrixB::Scalar,
      Scalar2, typename MatrixC::Scalar>::type;

   static value_type eval(Scalar1 alpha, const MatrixA& A, const MatrixB& B,
                          Scalar2 beta, MatrixC& C, Op_flag opA, Op_flag opB)
      {
         if (opA == Op_flag::None && opB == Op_flag::None) {
            return (alpha * A * B + beta * C).sum();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::None) {
            return (alpha * A.transpose() * B + beta * C).sum();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::None) {
            return (alpha * A.adjoint() * B + beta * C).sum();
         } else if (opA == Op_flag::None && opB == Op_flag::Transpose) {
            return (alpha * A * B.transpose() + beta * C).sum();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Transpose) {
            return (alpha * A.transpose() * B.transpose() + beta * C).sum();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Transpose) {
            return (alpha * A.adjoint() * B.transpose() + beta * C).sum();
         } else if (opA == Op_flag::None && opB == Op_flag::Adjoint) {
            return (alpha * A * B.adjoint() + beta * C).sum();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Adjoint) {
            return (alpha * A.transpose() * B.adjoint() + beta * C).sum();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Adjoint) {
            return (alpha * A.adjoint() * B.adjoint() + beta * C).sum();
         }

         throw std::runtime_error("invalid combination of operations");
      }
};

template <class Scalar1, class MatrixA, class MatrixB>
struct sum_gemm_op_impl<
   Scalar1, MatrixA, MatrixB,
   typename std::enable_if<is_eigen_matrix<MatrixA>::value &&
                           is_eigen_matrix<MatrixB>::value>::type> {

   using value_type = typename std::common_type<
      Scalar1, typename MatrixA::Scalar, typename MatrixB::Scalar>::type;

   static value_type eval(Scalar1 alpha, const MatrixA& A, const MatrixB& B,
                          Op_flag opA, Op_flag opB)
      {
         if (opA == Op_flag::None && opB == Op_flag::None) {
            return (alpha * A * B).sum();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::None) {
            return (alpha * A.transpose() * B).sum();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::None) {
            return (alpha * A.adjoint() * B).sum();
         } else if (opA == Op_flag::None && opB == Op_flag::Transpose) {
            return (alpha * A * B.transpose()).sum();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Transpose) {
            return (alpha * A.transpose() * B.transpose()).sum();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Transpose) {
            return (alpha * A.adjoint() * B.transpose()).sum();
         } else if (opA == Op_flag::None && opB == Op_flag::Adjoint) {
            return (alpha * A * B.adjoint()).sum();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Adjoint) {
            return (alpha * A.transpose() * B.adjoint()).sum();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Adjoint) {
            return (alpha * A.adjoint() * B.adjoint()).sum();
         }

         throw std::runtime_error("invalid combination of operations");
      }
};

template <class Scalar1, class MatrixA, class MatrixB, class Scalar2,
          class MatrixC>
struct hadamard_impl<
   Scalar1, MatrixA, MatrixB, Scalar2, MatrixC,
   typename std::enable_if<is_eigen_matrix<MatrixA>::value &&
                           is_eigen_matrix<MatrixB>::value &&
                           is_eigen_matrix<MatrixC>::value>::type> {

   static void eval(Scalar1 alpha, const MatrixA& A, const MatrixB& B,
                    Scalar2 beta, MatrixC& C, Op_flag opA, Op_flag opB)
      {
         if (opA == Op_flag::None && opB == Op_flag::None) {
            C = alpha * A.cwiseProduct(B) + beta * C;
         } else if (opA == Op_flag::Transpose && opB == Op_flag::None) {
            C = alpha * A.transpose().cwiseProduct(B) + beta * C;
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::None) {
            C = alpha * A.adjoint().cwiseProduct(B) + beta * C;
         } else if (opA == Op_flag::None && opB == Op_flag::Transpose) {
            C = alpha * A.cwiseProduct(B.transpose()) + beta * C;
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Transpose) {
            C = alpha * A.transpose().cwiseProduct(B.transpose()) + beta * C;
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Transpose) {
            C = alpha * A.adjoint().cwiseProduct(B.transpose()) + beta * C;
         } else if (opA == Op_flag::None && opB == Op_flag::Adjoint) {
            C = alpha * A.cwiseProduct(B.adjoint()) + beta * C;
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Adjoint) {
            C = alpha * A.transpose().cwiseProduct(B.adjoint()) + beta * C;
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Adjoint) {
            C = alpha * A.adjoint().cwiseProduct(B.adjoint()) + beta * C;
         }
      }
};

template <class Scalar1, class MatrixA, class MatrixB>
struct sum_hadamard_op_impl<
   Scalar1, MatrixA, MatrixB,
   typename std::enable_if<is_eigen_matrix<MatrixA>::value &&
                           is_eigen_matrix<MatrixB>::value>::type> {

   using value_type = typename std::common_type<
      Scalar1, typename MatrixA::Scalar, typename MatrixB::Scalar>::type;

   static value_type eval(Scalar1 alpha, const MatrixA& A, const MatrixB& B,
                          Op_flag opA, Op_flag opB)
      {
         if (opA == Op_flag::None && opB == Op_flag::None) {
            return (alpha * A.cwiseProduct(B)).sum();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::None) {
            return (alpha * A.transpose().cwiseProduct(B)).sum();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::None) {
            return (alpha * A.adjoint().cwiseProduct(B)).sum();
         } else if (opA == Op_flag::None && opB == Op_flag::Transpose) {
            return (alpha * A.cwiseProduct(B.transpose())).sum();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Transpose) {
            return (alpha * A.transpose().cwiseProduct(B.transpose())).sum();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Transpose) {
            return (alpha * A.adjoint().cwiseProduct(B.transpose())).sum();
         } else if (opA == Op_flag::None && opB == Op_flag::Adjoint) {
            return (alpha * A.cwiseProduct(B.adjoint())).sum();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Adjoint) {
            return (alpha * A.transpose().cwiseProduct(B.adjoint())).sum();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Adjoint) {
            return (alpha * A.adjoint().cwiseProduct(B.adjoint())).sum();
         }

         throw std::runtime_error("invalid combination of operations");
      }
};

template <class MatrixC, class MatrixA, class MatrixB, class ResidualMatrix>
struct matrix_residual_impl<
   MatrixC, MatrixA, MatrixB, ResidualMatrix,
   typename std::enable_if<is_eigen_matrix<MatrixC>::value &&
                           is_eigen_matrix<MatrixA>::value &&
                           is_eigen_matrix<MatrixB>::value &&
                           is_eigen_matrix<ResidualMatrix>::value>::type> {

   static void eval(const MatrixC& C, const MatrixA& A, const MatrixB& B,
                    ResidualMatrix& res)
      {
         res = C - A * B;
      }
};

template <class MatrixC, class MatrixA, class MatrixB>
struct matrix_residual_fro_norm_impl<
   MatrixC, MatrixA, MatrixB,
   typename std::enable_if<is_eigen_matrix<MatrixC>::value &&
                           is_eigen_matrix<MatrixA>::value &&
                           is_eigen_matrix<MatrixB>::value>::type> {
   using value_type = typename std::common_type<typename MatrixC::Scalar,
                                                typename MatrixA::Scalar,
                                                typename MatrixB::Scalar>::type;

   static value_type eval(const MatrixC& C, const MatrixA& A, const MatrixB& B)
      {
         return (C - A * B).norm();
      }
};

template <class MatrixA, class MatrixB>
struct solve_square_qr_left_impl<
   MatrixA, MatrixB,
   typename std::enable_if<is_eigen_matrix<MatrixA>::value &&
                           is_eigen_matrix<MatrixB>::value>::type> {

   using return_type = int;

   static return_type eval(const MatrixA& A, MatrixB& B, Op_flag opA)
      {
         if (A.rows() != A.cols()) {
            throw std::runtime_error("matrix A must be square");
         }

         switch (opA) {
         case Op_flag::None: {
            B = A.colPivHouseholderQr().solve(B);
            break;
         }
         case Op_flag::Transpose: {
            B = A.transpose().colPivHouseholderQr().solve(B);
            break;
         }
         case Op_flag::Adjoint: {
            B = A.adjoint().colPivHouseholderQr().solve(B);
            break;
         }
         }

         return 0;
      }
};

template <class MatrixA, class MatrixB>
struct solve_square_qr_right_impl<
   MatrixA, MatrixB,
   typename std::enable_if<is_eigen_matrix<MatrixA>::value &&
                           is_eigen_matrix<MatrixB>::value>::type> {

   using return_type = int;

   static return_type eval(const MatrixA& A, MatrixB& B, Op_flag opA)
      {
         if (A.rows() != A.cols()) {
            throw std::runtime_error("matrix A must be square");
         }

         B.transposeInPlace();

         switch (opA) {
         case Op_flag::None: {
            B = A.transpose().colPivHouseholderQr().solve(B);
            break;
         }
         case Op_flag::Transpose: {
            B = A.colPivHouseholderQr().solve(B);
            break;
         }
         case Op_flag::Adjoint: {
            B = A.conjugate().colPivHouseholderQr().solve(B);
            break;
         }
         }

         B.transposeInPlace();

         return 0;
      }
};

template <class Vector>
struct simplex_project_vector_impl<
   Vector,
   typename std::enable_if<is_eigen_matrix<Vector>::value>::type> {

   using Index = Eigen::Index;
   using value_type = typename Vector::Scalar;

   static void eval(Vector& x)
      {
         using std::max;
         using std::sort;

         Vector sorted_x(x);
         sort(sorted_x.data(), sorted_x.data() + sorted_x.size());

         const Index n = x.size();

         value_type t_hat = 0;
         for (Index i = n - 2; i >= -1; --i) {
            t_hat = (sorted_x.tail(n - 1 - i).sum() - 1) / (n - 1 - i);
            if (t_hat >= sorted_x(i)) {
               break;
            }
         }

         std::for_each(
            x.data(), x.data() + x.size(),
            [t_hat](double& xi) { xi = max(xi - t_hat, 0.0); });
      }
};

template <class Matrix>
struct simplex_project_columns_impl<
   Matrix,
   typename std::enable_if<is_eigen_matrix<Matrix>::value>::type> {

   using Index = Eigen::Index;

   static void eval(Matrix& A)
      {
         const Index n_cols = A.cols();
         for (Index i = 0; i < n_cols; ++i) {
            Eigen::VectorXd c(A.col(i));
            simplex_project_vector(c);
            A.col(i) = c;
         }
      }
};

} // namespace detail

} // namespace backends

} // namespace reor

#endif
