MODULE dlaf
  USE kinds, ONLY: dp
  USE ISO_C_BINDING, ONLY: C_PTR, C_INT, C_DOUBLE, C_CHAR, C_FLOAT, C_LOC

#include "../base/base_uses.f90"
  
  IMPLICIT NONE
  PRIVATE
  CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'cp_dlaf'
  PUBLIC :: cp_pdpotrf_dlaf, cp_pspotrf_dlaf, cp_dlaf_init, cp_dlaf_finalize
CONTAINS

      SUBROUTINE cp_dlaf_init()
        INTERFACE
           SUBROUTINE dlaf_init_aux() &
                BIND(C, name='dlaf_init')
           end SUBROUTINE dlaf_init_aux
        end INTERFACE

        !character(:), allocatable :: arg
        !integer arglen, stat
        !call get_command_argument(number=1, length=arglen)  ! Assume for simplicity success
        !allocate (character(arglen) :: arg)
        !call get_command_argument(number=1, value=arg, status=stat)

        call dlaf_init_aux()
      end SUBROUTINE cp_dlaf_init

      SUBROUTINE cp_dlaf_finalize()
        INTERFACE
           SUBROUTINE dlaf_finalize_aux() &
                BIND(C, name='dlaf_finalize')
           end SUBROUTINE dlaf_finalize_aux
        end INTERFACE
        call dlaf_finalize_aux()
      end SUBROUTINE cp_dlaf_finalize

  SUBROUTINE cp_pdpotrf_dlaf(uplo, n, a, ia, ja, desca, info)
    CHARACTER, INTENT(IN) ::           uplo
    INTEGER, INTENT(IN)       ::       ia, ja, n
    INTEGER, TARGET     ::      info
    INTEGER, dimension(:), TARGET       ::        DESCA
    REAL(KIND=dp), DIMENSION(:, :), TARGET   :: A
    INTERFACE
       SUBROUTINE pdpotrf_dlaf(UPLO, N, A, IA, JA, DESCA, INFO) &
         BIND(C, name='pdpotrf_dlaf')
         IMPORT :: C_PTR, C_INT, C_DOUBLE, C_CHAR
         CHARACTER(KIND=C_CHAR), VALUE :: UPLO
         INTEGER(KIND=C_INT), VALUE      ::        IA, JA, N
         TYPE(C_PTR), value ::       INFO
         TYPE(C_PTR), value ::        DESCA
         TYPE(C_PTR), value ::              A
       END SUBROUTINE pdpotrf_dlaf
    END INTERFACE
#if __GNUC__ >= 9
    CPASSERT(IS_CONTIGUOUS(A))
#endif
    CALL pdpotrf_dlaf(UPLO, N, C_LOC(A(1, 1)), IA, JA, C_LOC(DESCA(1)), C_LOC(INFO))
  END SUBROUTINE cp_pdpotrf_dlaf

  SUBROUTINE cp_pspotrf_dlaf(UPLO, N, A, IA, JA, DESCA, INFO) 
    CHARACTER, INTENT(IN) ::           UPLO
    INTEGER, INTENT(IN)       ::       IA, JA, N
    INTEGER, TARGET     ::      INFO
    INTEGER, dimension(:), TARGET      ::        DESCA
    REAL, dimension(:,:), TARGET    ::              A
    INTERFACE
       SUBROUTINE pspotrf_dlaf(UPLO, N, A, IA, JA, DESCA, INFO) &
         BIND(C, name='pspotrf_dlaf')
         IMPORT :: C_PTR, C_INT, C_DOUBLE, C_CHAR
         CHARACTER(KIND=C_CHAR), VALUE :: UPLO
         INTEGER(KIND=C_INT), value      ::        IA, JA, N
         TYPE(C_PTR), value    ::       INFO
         TYPE(C_PTR), value      ::        DESCA
         TYPE(C_PTR), value   ::              A
       END SUBROUTINE pspotrf_dlaf
    END INTERFACE
    CALL pspotrf_dlaf(UPLO, N,  C_LOC(A(1, 1)), IA, JA, C_LOC(DESCA(1)), C_LOC(INFO))
  END SUBROUTINE cp_pspotrf_dlaf
END MODULE dlaf

