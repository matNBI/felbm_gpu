#ifndef FELBM_GPU_OPERATOR_ACCESS_H
#define FELBM_GPU_OPERATOR_ACCESS_H

//=============================================================================
//  Thin subclasses that expose the CPU operators' precomputed CSR arrays.
//
//  FieldOperator / StreamingOperator keep their CSR members `protected`, so a
//  DERIVED class can read them without any edit to felbm_local. This keeps
//  felbm_gpu fully self-contained: it works against an unmodified felbm_local
//  checkout (no accessors need to be added upstream).
//
//  C++14 `auto const&` return-type deduction gives back the exact member types
//  (std::vector<index_type> / std::vector<double>); device_csr.cuh's upload()
//  is templated so it accepts them whether index_type is int or unsigned int.
//=============================================================================

#include <lbm_field_operator.h>
#include <lbm_streaming_operator.h>

namespace felbm_gpu
{
  struct FieldOperatorGPU : public lbm::FieldOperator
  {
    FieldOperatorGPU( lbm::SubDomain const & d, lbm::VelocitySet const & v, lbm::Settings const & s )
    : lbm::FieldOperator( d, v, s ) {}

    auto const & rows_cd()      const { return m_rows_cd;       }
    auto const & cols_cd()      const { return m_cols_cd;       }
    auto const & grad_cd_x()    const { return m_grad_cd_x;     }
    auto const & grad_cd_y()    const { return m_grad_cd_y;     }
    auto const & grad_cd_z()    const { return m_grad_cd_z;     }

    auto const & rows_bd()      const { return m_rows_bd;       }
    auto const & cols_bd()      const { return m_cols_bd;       }
    auto const & grad_bd_x()    const { return m_grad_bd_x;     }
    auto const & grad_bd_y()    const { return m_grad_bd_y;     }
    auto const & grad_bd_z()    const { return m_grad_bd_z;     }

    auto const & rows_lap()     const { return m_rows_lap;      }
    auto const & cols_lap()     const { return m_cols_lap;      }
    auto const & laplacian()    const { return m_laplacian;     }

    auto const & rows_cd_dir()  const { return m_rows_cd_dir;   }
    auto const & cols_cd_dir()  const { return m_cols_cd_dir;   }
    auto const & grad_cd_dir()  const { return m_grad_cd_dir;   }

    auto const & rows_bd_dir()  const { return m_rows_bd_dir;   }
    auto const & cols_bd_dir()  const { return m_cols_bd_dir;   }
    auto const & grad_bd_dir()  const { return m_grad_bd_dir;   }

    auto const & rows_avg_dir() const { return m_rows_avg_dir;  }
    auto const & cols_avg_dir() const { return m_cols_avg_dir;  }
    auto const & values_avg_dir() const { return m_values_avg_dir; }
  };

  struct StreamingOperatorGPU : public lbm::StreamingOperator
  {
    StreamingOperatorGPU( lbm::SubDomain const & d, lbm::VelocitySet const & v, lbm::Settings const & s )
    : lbm::StreamingOperator( d, v, s ) {}

    auto const & rows()   const { return m_rows;   }
    auto const & cols()   const { return m_cols;   }
    auto const & values() const { return m_values; }
  };

} // namespace felbm_gpu

#endif // FELBM_GPU_OPERATOR_ACCESS_H
