! fused_bench.f90
! Fortran90/OpenMP counterpart to fused_bench.c (Paper II fused
! forward+backward, Algorithm 2, A never materialized).
!
! Usage: ./fused_bench_f90 B N D THREADS [--check]
! Prints: fused,Fortran,B,N,D,threads,seconds
program fused_bench
  use omp_lib
  implicit none
  integer :: B, N, D, THREADS, NT
  real(8), allocatable :: Q(:), K(:), V(:), dO(:), Out(:), GV(:), GQ(:), GK(:)
  real(8), allocatable :: A_ref(:), Out_ref(:), GV_ref(:), GQ_ref(:), GK_ref(:)
  real(8) :: t0, t1, scale
  integer :: b_i, ir, base_ir
  character(len=32) :: arg
  logical :: do_check

  if (command_argument_count() < 4) then
    print *, "usage: fused_bench_f90 B N D THREADS [--check]"
    stop 1
  end if
  call get_command_argument(1, arg); read(arg,*) B
  call get_command_argument(2, arg); read(arg,*) N
  call get_command_argument(3, arg); read(arg,*) D
  call get_command_argument(4, arg); read(arg,*) THREADS
  do_check = .false.
  if (command_argument_count() >= 5) then
    call get_command_argument(5, arg)
    if (trim(arg) == "--check") do_check = .true.
  end if

  call omp_set_num_threads(THREADS)
  scale = 1.0d0 / sqrt(dble(D))
  NT = B * N

  allocate(Q(NT*D), K(NT*D), V(NT*D), dO(NT*D))
  allocate(Out(NT*D), GV(NT*D), GQ(NT*D), GK(NT*D))
  call fill_rand(Q, NT*D, 1)
  call fill_rand(K, NT*D, 2)
  call fill_rand(V, NT*D, 3)
  call fill_rand(dO, NT*D, 4)
  Out = 0.0d0; GV = 0.0d0; GQ = 0.0d0; GK = 0.0d0

  t0 = omp_get_wtime()

  ! GV and GK accumulate across different ir values writing the same
  ! ic-indexed slot -- both need array reductions. GQ and Out need none.
  !$omp parallel do collapse(2) private(base_ir) reduction(+:GV, GK)
  do b_i = 0, B-1
    do ir = 0, N-1
      base_ir = (b_i*N+ir)*D
      block
        real(8) :: arow_l(N), ga_row(N), corr, gs, s, m, sm, vv, oo
        integer :: icc, jj, base_ic
        do icc = 0, N-1
          base_ic = (b_i*N+icc)*D
          s = 0.0d0
          do jj = 0, D-1
            s = s + Q(base_ir+jj+1) * K(base_ic+jj+1)
          end do
          arow_l(icc+1) = s * scale
        end do
        m = maxval(arow_l)
        arow_l = exp(arow_l - m)
        sm = sum(arow_l)
        arow_l = arow_l / sm

        do icc = 0, N-1
          base_ic = (b_i*N+icc)*D
          ga_row(icc+1) = 0.0d0
          do jj = 0, D-1
            vv = V(base_ic+jj+1)
            oo = dO(base_ir+jj+1)
            ga_row(icc+1) = ga_row(icc+1) + oo * vv
            Out(base_ir+jj+1) = Out(base_ir+jj+1) + arow_l(icc+1) * vv
            GV(base_ic+jj+1) = GV(base_ic+jj+1) + arow_l(icc+1) * oo
          end do
        end do

        corr = 0.0d0
        do icc = 0, N-1
          corr = corr + arow_l(icc+1) * ga_row(icc+1)
        end do
        do icc = 0, N-1
          gs = scale * arow_l(icc+1) * (ga_row(icc+1) - corr)
          base_ic = (b_i*N+icc)*D
          do jj = 0, D-1
            GQ(base_ir+jj+1) = GQ(base_ir+jj+1) + gs * K(base_ic+jj+1)
            GK(base_ic+jj+1) = GK(base_ic+jj+1) + gs * Q(base_ir+jj+1)
          end do
        end do
      end block
    end do
  end do
  !$omp end parallel do

  t1 = omp_get_wtime()

  print '(A,I0,A,I0,A,I0,A,I0,A,F10.6)', &
    "fused,Fortran,", B, ",", N, ",", D, ",", THREADS, ",", t1-t0

  if (do_check) then
    allocate(A_ref(NT*N), Out_ref(NT*D), GV_ref(NT*D), GQ_ref(NT*D), GK_ref(NT*D))
    Out_ref = 0.0d0; GV_ref = 0.0d0; GQ_ref = 0.0d0; GK_ref = 0.0d0
    call forward_serial(Q, K, V, A_ref, Out_ref, scale, B, N, D)
    call backward_serial(Q, K, V, A_ref, dO, GV_ref, GQ_ref, GK_ref, scale, B, N, D)
    print '(A,ES12.3)', "check: Out max diff = ", maxval(abs(Out - Out_ref))
    print '(A,ES12.3)', "check: GV  max diff = ", maxval(abs(GV - GV_ref))
    print '(A,ES12.3)', "check: GQ  max diff = ", maxval(abs(GQ - GQ_ref))
    print '(A,ES12.3)', "check: GK  max diff = ", maxval(abs(GK - GK_ref))
    deallocate(A_ref, Out_ref, GV_ref, GQ_ref, GK_ref)
  end if

  deallocate(Q, K, V, dO, Out, GV, GQ, GK)

contains

  subroutine forward_serial(Q, K, V, A, Out, scale, B, N, D)
    real(8), intent(in) :: Q(*), K(*), V(*), scale
    real(8), intent(out) :: A(*), Out(*)
    integer, intent(in) :: B, N, D
    integer :: b_i, ir, ic, id, jj, base_ir, base_ic
    real(8) :: arow_l(N), s, m, sm, o
    do b_i = 0, B-1
      do ir = 0, N-1
        base_ir = (b_i*N+ir)*D
        do ic = 0, N-1
          base_ic = (b_i*N+ic)*D
          s = 0.0d0
          do jj = 0, D-1
            s = s + Q(base_ir+jj+1) * K(base_ic+jj+1)
          end do
          arow_l(ic+1) = s * scale
        end do
        m = maxval(arow_l)
        arow_l = exp(arow_l - m)
        sm = sum(arow_l)
        arow_l = arow_l / sm
        do ic = 0, N-1
          A((b_i*N+ir)*N+ic+1) = arow_l(ic+1)
        end do
        do id = 0, D-1
          o = 0.0d0
          do ic = 0, N-1
            o = o + arow_l(ic+1) * V((b_i*N+ic)*D+id+1)
          end do
          Out(base_ir+id+1) = o
        end do
      end do
    end do
  end subroutine forward_serial

  subroutine backward_serial(Q, K, V, A, dO, GV, GQ, GK, scale, B, N, D)
    real(8), intent(in) :: Q(*), K(*), V(*), A(*), dO(*), scale
    real(8), intent(inout) :: GV(*), GQ(*), GK(*)
    integer, intent(in) :: B, N, D
    integer :: b_i, ir, ic, id, jj, base_ir, base_ic
    real(8) :: ga_row(N), corr, gs, g
    do b_i = 0, B-1
      do ic = 0, N-1
        base_ic = (b_i*N+ic)*D
        do id = 0, D-1
          g = 0.0d0
          do ir = 0, N-1
            g = g + A((b_i*N+ir)*N+ic+1) * dO((b_i*N+ir)*D+id+1)
          end do
          GV(base_ic+id+1) = g
        end do
      end do
      do ir = 0, N-1
        base_ir = (b_i*N+ir)*D
        do jj = 0, D-1
          GQ(base_ir+jj+1) = 0.0d0
        end do
      end do
      do ir = 0, N-1
        base_ir = (b_i*N+ir)*D
        do ic = 0, N-1
          ga_row(ic+1) = 0.0d0
          do jj = 0, D-1
            ga_row(ic+1) = ga_row(ic+1) + dO(base_ir+jj+1) * V((b_i*N+ic)*D+jj+1)
          end do
        end do
        corr = 0.0d0
        do ic = 0, N-1
          corr = corr + A((b_i*N+ir)*N+ic+1) * ga_row(ic+1)
        end do
        do ic = 0, N-1
          gs = scale * A((b_i*N+ir)*N+ic+1) * (ga_row(ic+1) - corr)
          do jj = 0, D-1
            GQ(base_ir+jj+1) = GQ(base_ir+jj+1) + gs * K((b_i*N+ic)*D+jj+1)
            GK((b_i*N+ic)*D+jj+1) = GK((b_i*N+ic)*D+jj+1) + gs * Q(base_ir+jj+1)
          end do
        end do
      end do
    end do
  end subroutine backward_serial

  subroutine fill_rand(arr, n, seed_in)
    real(8), intent(out) :: arr(n)
    integer, intent(in) :: n, seed_in
    integer(8) :: s
    integer :: i
    s = int(seed_in, 8)
    do i = 1, n
      s = mod(s * 1103515245_8 + 12345_8, 2147483648_8)
      arr(i) = (dble(mod(s, 20000_8)) / 10000.0d0 - 1.0d0)
    end do
  end subroutine fill_rand

end program fused_bench
