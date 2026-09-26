! Parameterized fused RMSNorm -> gated MLP forward+backward benchmark (Fortran90/OpenACC).
! Counterpart to fused_gpu_kernel.c, Sections 3-9 of the paper.
!
! REVISION HISTORY: the original version took a single N per process
! invocation, with the sweep script launching one process per N -- the
! same flaw diagnosed from real Delta A100 data on decode_gpu_bench.c
! (job 21180661): CUDA context creation costs ~300ms PER PROCESS
! LAUNCH, dominating and masking the actual kernel time at every N.
! This version sweeps N internally within a single process, paying
! context-init cost once via an explicit untimed warmup call before any
! measurement begins. Same fix as fused_gpu_bench.c (the C version) and
! every other *_gpu_bench.c file in this project.
!
! Usage: ./fused_gpu_bench_f90 B N_START N_END D DFF
! Sweeps N from N_START to N_END, doubling. Prints one line per N:
! fused,Fortran-OpenACC,B,N,D,DFF,seconds
!
! NOTE ON THIS ENVIRONMENT: compiled and timed here via OpenACC host
! fallback (no CUDA GPU / nvfortran toolchain in this environment) --
! every !$acc directive is standard OpenACC and should offload
! unmodified under nvfortran. This file has NOT itself been run on real
! GPU hardware yet.
program fused_gpu_bench
  implicit none
  integer :: B, N_START, N_END, D, DFF, N_cur
  character(len=32) :: arg
  real(8) :: elapsed

  if (command_argument_count() < 5) then
    print *, "usage: fused_gpu_bench_f90 B N_START N_END D DFF"
    stop 1
  end if
  call get_command_argument(1, arg); read(arg,*) B
  call get_command_argument(2, arg); read(arg,*) N_START
  call get_command_argument(3, arg); read(arg,*) N_END
  call get_command_argument(4, arg); read(arg,*) D
  call get_command_argument(5, arg); read(arg,*) DFF

  if (D > 4096 .or. DFF > 4096) then
    print *, "D or DFF exceeds fixed scratch buffer size (4096)"
    stop 1
  end if

  ! Untimed warmup: pays CUDA context creation / OpenACC JIT cost once,
  ! before any measurement begins, at a small fixed size.
  write(0,*) "warming up (paying CUDA context init cost once, untimed)..."
  call run_once(B, min(N_START, 64), D, DFF, elapsed)

  N_cur = N_START
  do while (N_cur <= N_END)
    call run_once(B, N_cur, D, DFF, elapsed)
    print '(A,I0,A,I0,A,I0,A,I0,A,F10.6)', &
      "fused,Fortran-OpenACC,", B, ",", N_cur, ",", D, ",", DFF, ",", elapsed
    N_cur = N_cur * 2
  end do

contains

  subroutine run_once(B, N, D, DFF, elapsed)
    integer, intent(in) :: B, N, D, DFF
    real(8), intent(out) :: elapsed
    integer :: NT
    real(8), allocatable :: X(:), gam(:), GF(:), Z(:), U(:), Fout(:)
    real(8), allocatable :: Wg(:), Wu(:), Wd(:), r(:)
    real(8), allocatable :: GX_norm(:), Ggamma(:), GX_ffn(:), GWg(:), GWu(:), GWd(:)
    real(8) :: EPS, sumsq, u_val, v_val, f_val
    real(8) :: Vrow(4096), Hrow(4096), GHrow(4096), GUrow(4096), GVrow(4096), GZf(4096)
    real(8) :: gh, gz_val, correction, x_hat_d, x_hat_e
    integer :: t_i, d_i, f_i, e_i, xbase, ubase, wgbase, wdbase
    integer :: count_rate, count0, count1

    EPS = 1.0d-6
    NT = B * N
    allocate(X(NT*D), gam(D), GF(NT*D), Z(NT*D), U(NT*DFF), Fout(NT*D))
    allocate(Wg(D*DFF), Wu(D*DFF), Wd(DFF*D), r(NT))
    allocate(GX_norm(NT*D), Ggamma(D), GX_ffn(NT*D), GWg(D*DFF), GWu(D*DFF), GWd(DFF*D))

    call fill_rand(X, NT*D, 1)
    call fill_rand(gam, D, 2)
    call fill_rand(GF, NT*D, 15)
    call fill_rand(Wg, D*DFF, 12)
    call fill_rand(Wu, D*DFF, 13)
    call fill_rand(Wd, DFF*D, 14)
    Ggamma = 0.0d0

    call system_clock(count0, count_rate)

    !$acc data copyin(X, gam, GF, Wg, Wu, Wd) &
    !$acc      copyout(Z, Fout, GX_norm, Ggamma, GX_ffn, GWg, GWu, GWd) &
    !$acc      create(U, r)

    ! ---- RMSNorm forward (Section 3) ----
    !$acc parallel loop private(sumsq, d_i)
    do t_i = 0, NT-1
      sumsq = 0.0d0
      do d_i = 1, D
        sumsq = sumsq + X(t_i*D+d_i)*X(t_i*D+d_i)
      end do
      r(t_i+1) = sqrt(sumsq/dble(D) + EPS)
    end do
    !$acc end parallel loop

    !$acc parallel loop collapse(2)
    do t_i = 0, NT-1
      do d_i = 1, D
        Z(t_i*D+d_i) = gam(d_i) * (X(t_i*D+d_i) / r(t_i+1))
      end do
    end do
    !$acc end parallel loop

    ! ---- Gated MLP forward (Section 6), input Z ----
    !$acc parallel loop private(xbase, ubase, d_i, f_i, u_val, v_val, f_val, wgbase, wdbase, Vrow, Hrow)
    do t_i = 0, NT-1
      xbase = t_i*D
      ubase = t_i*DFF
      do f_i = 0, DFF-1
        u_val = 0.0d0
        v_val = 0.0d0
        do d_i = 0, D-1
          wgbase = d_i*DFF
          u_val = u_val + Z(xbase+d_i+1) * Wg(wgbase+f_i+1)
          v_val = v_val + Z(xbase+d_i+1) * Wu(wgbase+f_i+1)
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
    !$acc end parallel loop

    ! ---- Gated MLP backward (Sections 6-7) + RMSNorm backward (Section 4), fused ----
    ! Every "transpose" access (Wd, Wg, Wu, X, H) below is a reordered-index read,
    ! never a materialized transposed array (Section 8 lemma).
    !$acc parallel loop private(xbase, ubase, d_i, f_i, e_i, v_val, gh, gz_val, correction, &
    !$acc                       x_hat_d, x_hat_e, wgbase, wdbase, Vrow, Hrow, GHrow, GUrow, GVrow, GZf)
    do t_i = 0, NT-1
      xbase = t_i*D
      ubase = t_i*DFF
      do f_i = 0, DFF-1
        v_val = 0.0d0
        do d_i = 0, D-1
          wgbase = d_i*DFF
          v_val = v_val + Z(xbase+d_i+1) * Wu(wgbase+f_i+1)
        end do
        Vrow(f_i+1) = v_val
        Hrow(f_i+1) = silu(U(ubase+f_i+1)) * v_val
      end do
      do f_i = 0, DFF-1
        wdbase = f_i*D
        do d_i = 0, D-1
          !$acc atomic update
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
          !$acc atomic update
          GWg(wgbase+f_i+1) = GWg(wgbase+f_i+1) + Z(xbase+d_i+1) * GUrow(f_i+1)
          !$acc atomic update
          GWu(wgbase+f_i+1) = GWu(wgbase+f_i+1) + Z(xbase+d_i+1) * GVrow(f_i+1)
        end do
      end do
      do d_i = 0, D-1
        wgbase = d_i*DFF
        gz_val = 0.0d0
        do f_i = 0, DFF-1
          gz_val = gz_val + GUrow(f_i+1)*Wg(wgbase+f_i+1) + GVrow(f_i+1)*Wu(wgbase+f_i+1)
        end do
        GZf(d_i+1) = gz_val
        GX_ffn(xbase+d_i+1) = gz_val
      end do
      ! RMSNorm backward, x_hat recomputed on-device, fused in the same kernel launch
      correction = 0.0d0
      do d_i = 1, D
        x_hat_d = X(xbase+d_i) / r(t_i+1)
        correction = correction + GZf(d_i) * gam(d_i) * x_hat_d
      end do
      do e_i = 1, D
        x_hat_e = X(xbase+e_i) / r(t_i+1)
        GX_norm(xbase+e_i) = (GZf(e_i)*gam(e_i) - (x_hat_e/dble(D))*correction) / r(t_i+1)
        !$acc atomic update
        Ggamma(e_i) = Ggamma(e_i) + GZf(e_i) * x_hat_e
      end do
    end do
    !$acc end parallel loop

    !$acc end data

    call system_clock(count1)
    elapsed = dble(count1 - count0) / dble(count_rate)

    deallocate(X, gam, GF, Z, U, Fout, Wg, Wu, Wd, r)
    deallocate(GX_norm, Ggamma, GX_ffn, GWg, GWu, GWd)
  end subroutine run_once

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

end program fused_gpu_bench
