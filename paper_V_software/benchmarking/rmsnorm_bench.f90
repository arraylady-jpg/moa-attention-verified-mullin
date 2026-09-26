program rmsnorm_bench
  use omp_lib
  implicit none
  integer :: B, N, D, THREADS, NT
  real(8), allocatable :: X(:), gamma(:), GZ(:), Z(:), GX(:), Ggamma(:), r(:)
  real(8) :: EPS, t0, t1, sumsq, x_hat_d, x_hat_e, correction
  integer :: t_i, d_i, e_i, base
  character(len=32) :: arg
  integer :: seed

  EPS = 1.0d-6

  if (command_argument_count() < 4) then
    print *, "usage: rmsnorm_bench_f90 B N D THREADS"
    stop 1
  end if
  call get_command_argument(1, arg); read(arg,*) B
  call get_command_argument(2, arg); read(arg,*) N
  call get_command_argument(3, arg); read(arg,*) D
  call get_command_argument(4, arg); read(arg,*) THREADS

  call omp_set_num_threads(THREADS)

  NT = B * N
  allocate(X(NT*D), gamma(D), GZ(NT*D), Z(NT*D), GX(NT*D), Ggamma(D), r(NT))

  seed = 12345
  call fill_rand(X, NT*D, 1)
  call fill_rand(gamma, D, 2)
  call fill_rand(GZ, NT*D, 3)
  Ggamma = 0.0d0

  t0 = omp_get_wtime()

  !$omp parallel do private(base, d_i, sumsq)
  do t_i = 0, NT-1
    base = t_i*D
    sumsq = 0.0d0
    do d_i = 1, D
      sumsq = sumsq + X(base+d_i)*X(base+d_i)
    end do
    r(t_i+1) = sqrt(sumsq/dble(D) + EPS)
  end do
  !$omp end parallel do

  !$omp parallel do collapse(2) private(base)
  do t_i = 0, NT-1
    do d_i = 1, D
      base = t_i*D
      Z(base+d_i) = gamma(d_i) * (X(base+d_i) / r(t_i+1))
    end do
  end do
  !$omp end parallel do

  !$omp parallel do private(base, correction, x_hat_d, x_hat_e, d_i, e_i)
  do t_i = 0, NT-1
    base = t_i*D
    correction = 0.0d0
    do d_i = 1, D
      x_hat_d = X(base+d_i) / r(t_i+1)
      correction = correction + GZ(base+d_i) * gamma(d_i) * x_hat_d
    end do
    do e_i = 1, D
      x_hat_e = X(base+e_i) / r(t_i+1)
      GX(base+e_i) = (GZ(base+e_i)*gamma(e_i) - (x_hat_e/dble(D))*correction) / r(t_i+1)
    end do
  end do
  !$omp end parallel do

  !$omp parallel do private(base, x_hat_d, d_i) reduction(+:Ggamma)
  do t_i = 0, NT-1
    base = t_i*D
    do d_i = 1, D
      x_hat_d = X(base+d_i) / r(t_i+1)
      Ggamma(d_i) = Ggamma(d_i) + GZ(base+d_i) * x_hat_d
    end do
  end do
  !$omp end parallel do

  t1 = omp_get_wtime()

  print '(A,I0,A,I0,A,I0,A,I0,A,F10.6)', &
    "rmsnorm,Fortran,", B, ",", N, ",", D, ",0,", THREADS, ",", t1-t0

  deallocate(X, gamma, GZ, Z, GX, Ggamma, r)

contains

  subroutine fill_rand(arr, n, seed_in)
    real(8), intent(out) :: arr(n)
    integer, intent(in) :: n, seed_in
    integer(8) :: s
    integer :: i
    s = int(seed_in, 8)
    do i = 1, n
      s = mod(s * 1103515245_8 + 12345_8, 2147483648_8)
      arr(i) = (dble(mod(s, 20000_8)) / 10000.0d0 - 1.0d0) * 0.5d0
    end do
  end subroutine fill_rand

end program rmsnorm_bench
