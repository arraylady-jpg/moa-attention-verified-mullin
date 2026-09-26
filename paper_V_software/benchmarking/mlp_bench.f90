program mlp_bench
  use omp_lib
  implicit none
  integer :: B, N, D, DFF, THREADS, NT
  real(8), allocatable :: X(:), Wg(:), Wu(:), Wd(:), GF(:), U(:), Fout(:)
  real(8), allocatable :: GX(:), GWg(:), GWu(:), GWd(:)
  real(8) :: t0, t1, u_val, v_val, f_val, gh, gx_val
  integer :: t_i, d_i, f_i, xbase, ubase, wgbase, wdbase
  real(8) :: Vrow(4096), Hrow(4096), GHrow(4096), GUrow(4096), GVrow(4096)
  character(len=32) :: arg

  if (command_argument_count() < 5) then
    print *, "usage: mlp_bench_f90 B N D DFF THREADS"
    stop 1
  end if
  call get_command_argument(1, arg); read(arg,*) B
  call get_command_argument(2, arg); read(arg,*) N
  call get_command_argument(3, arg); read(arg,*) D
  call get_command_argument(4, arg); read(arg,*) DFF
  call get_command_argument(5, arg); read(arg,*) THREADS

  if (DFF > 4096) then
    print *, "DFF exceeds fixed scratch buffer size (4096); increase Vrow/Hrow/etc. bound."
    stop 1
  end if

  call omp_set_num_threads(THREADS)

  NT = B * N
  allocate(X(NT*D), Wg(D*DFF), Wu(D*DFF), Wd(DFF*D), GF(NT*D))
  allocate(U(NT*DFF), Fout(NT*D))
  allocate(GX(NT*D), GWg(D*DFF), GWu(D*DFF), GWd(DFF*D))

  call fill_rand(X, NT*D, 11)
  call fill_rand(Wg, D*DFF, 12)
  call fill_rand(Wu, D*DFF, 13)
  call fill_rand(Wd, DFF*D, 14)
  call fill_rand(GF, NT*D, 15)
  GWg = 0.0d0; GWu = 0.0d0; GWd = 0.0d0

  t0 = omp_get_wtime()

  !$omp parallel do private(xbase, ubase, d_i, f_i, u_val, v_val, f_val, wgbase, wdbase, Vrow, Hrow)
  do t_i = 0, NT-1
    xbase = t_i*D
    ubase = t_i*DFF
    do f_i = 0, DFF-1
      u_val = 0.0d0
      v_val = 0.0d0
      do d_i = 0, D-1
        wgbase = d_i*DFF
        u_val = u_val + X(xbase+d_i+1) * Wg(wgbase+f_i+1)
        v_val = v_val + X(xbase+d_i+1) * Wu(wgbase+f_i+1)
      end do
      U(ubase+f_i+1) = u_val
      Vrow(f_i+1) = v_val
      Hrow(f_i+1) = silu(u_val) * v_val
    end do
    do d_i = 0, D-1
      f_val = 0.0d0
      do f_i = 0, DFF-1
        wdbase = f_i*D
        f_val = f_val + Hrow(f_i+1) * Wd(wdbase+d_i+1)
      end do
      Fout(xbase+d_i+1) = f_val
    end do
  end do
  !$omp end parallel do

  !$omp parallel do private(xbase, ubase, d_i, f_i, v_val, gh, gx_val, wgbase, wdbase, &
  !$omp                     Vrow, Hrow, GHrow, GUrow, GVrow) reduction(+:GWg, GWu, GWd)
  do t_i = 0, NT-1
    xbase = t_i*D
    ubase = t_i*DFF
    do f_i = 0, DFF-1
      v_val = 0.0d0
      do d_i = 0, D-1
        wgbase = d_i*DFF
        v_val = v_val + X(xbase+d_i+1) * Wu(wgbase+f_i+1)
      end do
      Vrow(f_i+1) = v_val
      Hrow(f_i+1) = silu(U(ubase+f_i+1)) * v_val
    end do
    do f_i = 0, DFF-1
      wdbase = f_i*D
      do d_i = 0, D-1
        GWd(wdbase+d_i+1) = GWd(wdbase+d_i+1) + Hrow(f_i+1) * GF(xbase+d_i+1)
      end do
    end do
    do f_i = 0, DFF-1
      gh = 0.0d0
      wdbase = f_i*D
      do d_i = 0, D-1
        gh = gh + GF(xbase+d_i+1) * Wd(wdbase+d_i+1)
      end do
      GHrow(f_i+1) = gh
      GVrow(f_i+1) = gh * silu(U(ubase+f_i+1))
      GUrow(f_i+1) = gh * Vrow(f_i+1) * silu_grad(U(ubase+f_i+1))
    end do
    do d_i = 0, D-1
      wgbase = d_i*DFF
      do f_i = 0, DFF-1
        GWg(wgbase+f_i+1) = GWg(wgbase+f_i+1) + X(xbase+d_i+1) * GUrow(f_i+1)
        GWu(wgbase+f_i+1) = GWu(wgbase+f_i+1) + X(xbase+d_i+1) * GVrow(f_i+1)
      end do
    end do
    do d_i = 0, D-1
      wgbase = d_i*DFF
      gx_val = 0.0d0
      do f_i = 0, DFF-1
        gx_val = gx_val + GUrow(f_i+1)*Wg(wgbase+f_i+1) + GVrow(f_i+1)*Wu(wgbase+f_i+1)
      end do
      GX(xbase+d_i+1) = gx_val
    end do
  end do
  !$omp end parallel do

  t1 = omp_get_wtime()

  print '(A,I0,A,I0,A,I0,A,I0,A,I0,A,F10.6)', &
    "mlp,Fortran,", B, ",", N, ",", D, ",", DFF, ",", THREADS, ",", t1-t0

  deallocate(X, Wg, Wu, Wd, GF, U, Fout, GX, GWg, GWu, GWd)

contains

  function silu(u) result(res)
    real(8), intent(in) :: u
    real(8) :: res
    res = u / (1.0d0 + exp(-u))
  end function silu

  function silu_grad(u) result(res)
    real(8), intent(in) :: u
    real(8) :: res, s
    s = 1.0d0 / (1.0d0 + exp(-u))
    res = s * (1.0d0 + u * (1.0d0 - s))
  end function silu_grad

  subroutine fill_rand(arr, n, seed_in)
    real(8), intent(out) :: arr(n)
    integer, intent(in) :: n, seed_in
    integer(8) :: s
    integer :: i
    s = int(seed_in, 8)
    do i = 1, n
      s = mod(s * 1103515245_8 + 12345_8, 2147483648_8)
      arr(i) = (dble(mod(s, 20000_8)) / 10000.0d0 - 1.0d0) * 0.3d0
    end do
  end subroutine fill_rand

end program mlp_bench
