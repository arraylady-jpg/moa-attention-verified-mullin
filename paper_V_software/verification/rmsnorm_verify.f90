program rmsnorm_verify
  implicit none
  integer, parameter :: B = 2, N = 4, D = 8
  real(8), parameter :: EPS = 1.0d-6
  real(8) :: X(D*N*B), gamma(D), GZ(D*N*B)
  real(8) :: Z(D*N*B), GX(D*N*B), Ggamma(D), r(N*B)
  real(8) :: sumsq, x_hat_d, x_hat_e, correction
  integer :: b_i, n_i, d_i, e_i, base, ridx

  ! Flat index convention matches numpy row-major (B,N,D): base = ((b*N)+n)*D, +d, 0-based.
  ! Fortran loops are 1-based; b_i,n_i,d_i in 0..B-1/N-1/D-1 mirror python 0-based indices directly.

  call load_flat("data/X.bin", X, D*N*B)
  call load_flat("data/gamma.bin", gamma, D)
  call load_flat("data/GZ.bin", GZ, D*N*B)

  ! ---- Forward DNF (Section 3): reduction pass, then elementwise pass ----
  !$omp parallel do collapse(2) private(ridx, base, sumsq, d_i)
  do b_i = 0, B-1
    do n_i = 0, N-1
      ridx = b_i*N + n_i + 1
      base = (b_i*N + n_i) * D
      sumsq = 0.0d0
      do d_i = 0, D-1
        sumsq = sumsq + X(base+d_i+1) * X(base+d_i+1)
      end do
      r(ridx) = sqrt(sumsq / dble(D) + EPS)
    end do
  end do
  !$omp end parallel do

  !$omp parallel do collapse(2) private(ridx, base, d_i)
  do b_i = 0, B-1
    do n_i = 0, N-1
      ridx = b_i*N + n_i + 1
      base = (b_i*N + n_i) * D
      do d_i = 0, D-1
        Z(base+d_i+1) = gamma(d_i+1) * (X(base+d_i+1) / r(ridx))
      end do
    end do
  end do
  !$omp end parallel do

  call save_flat("data/f_Z.bin", Z, D*N*B)

  ! ---- Backward DNF (Section 4): x_hat recomputed, never stored ----
  Ggamma = 0.0d0
  !$omp parallel do collapse(2) private(ridx, base, correction, x_hat_d, x_hat_e, d_i, e_i)
  do b_i = 0, B-1
    do n_i = 0, N-1
      ridx = b_i*N + n_i + 1
      base = (b_i*N + n_i) * D
      correction = 0.0d0
      do d_i = 0, D-1
        x_hat_d = X(base+d_i+1) / r(ridx)   ! recomputed, not stored
        correction = correction + GZ(base+d_i+1) * gamma(d_i+1) * x_hat_d
      end do
      do e_i = 0, D-1
        x_hat_e = X(base+e_i+1) / r(ridx)   ! recomputed again
        GX(base+e_i+1) = (GZ(base+e_i+1) * gamma(e_i+1) &
                           - (x_hat_e / dble(D)) * correction) / r(ridx)
      end do
    end do
  end do
  !$omp end parallel do

  !$omp parallel do collapse(2) private(ridx, base, x_hat_d, d_i) reduction(+:Ggamma)
  do b_i = 0, B-1
    do n_i = 0, N-1
      ridx = b_i*N + n_i + 1
      base = (b_i*N + n_i) * D
      do d_i = 0, D-1
        x_hat_d = X(base+d_i+1) / r(ridx)
        Ggamma(d_i+1) = Ggamma(d_i+1) + GZ(base+d_i+1) * x_hat_d
      end do
    end do
  end do
  !$omp end parallel do

  call save_flat("data/f_GX_norm.bin", GX, D*N*B)
  call save_flat("data/f_Ggamma.bin", Ggamma, D)

  print *, "RMSNorm forward+backward complete. Z(1..4) = ", Z(1), Z(2), Z(3), Z(4)

contains

  subroutine load_flat(path, arr, count)
    character(len=*), intent(in) :: path
    integer, intent(in) :: count
    real(8), intent(out) :: arr(count)
    integer :: iu
    open(newunit=iu, file=path, access="stream", form="unformatted", status="old")
    read(iu) arr
    close(iu)
  end subroutine load_flat

  subroutine save_flat(path, arr, count)
    character(len=*), intent(in) :: path
    integer, intent(in) :: count
    real(8), intent(in) :: arr(count)
    integer :: iu
    open(newunit=iu, file=path, access="stream", form="unformatted", status="replace")
    write(iu) arr
    close(iu)
  end subroutine save_flat

end program rmsnorm_verify
