! backward_bench.f90
! Fortran90/OpenMP counterpart to backward_bench.c (Paper II backward
! pass, Algorithm 1, A from DRAM). Same math, same flat-array indexing
! convention as Paper IV's Fortran kernels.
!
! Usage: ./backward_bench_f90 B N D THREADS [--check]
! Prints: backward,Fortran,B,N,D,threads,seconds
program backward_bench
  use omp_lib
  implicit none
  integer :: B, N, D, THREADS, NT
  real(8), allocatable :: Q(:), K(:), V(:), dO(:), A(:), GV(:), GQ(:), GK(:)
  real(8), allocatable :: GV_ref(:), GQ_ref(:), GK_ref(:)
  real(8) :: t0, t1, scale
  integer :: b_i, ir, ic, id, base_ir, base_ic
  character(len=32) :: arg
  logical :: do_check

  if (command_argument_count() < 4) then
    print *, "usage: backward_bench_f90 B N D THREADS [--check]"
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

  allocate(Q(NT*D), K(NT*D), V(NT*D), dO(NT*D), A(NT*N))
  allocate(GV(NT*D), GQ(NT*D), GK(NT*D))
  call fill_rand(Q, NT*D, 1)
  call fill_rand(K, NT*D, 2)
  call fill_rand(V, NT*D, 3)
  call fill_rand(dO, NT*D, 4)
  GV = 0.0d0; GQ = 0.0d0; GK = 0.0d0

  ! Forward is a prerequisite (Algorithm 1 reads A "from DRAM"), not timed.
  call forward_serial(Q, K, V, A, scale, B, N, D)

  t0 = omp_get_wtime()

  ! GV[b,ic,id] = sum_ir A[b,ir,ic]*dO[b,ir,id] -- parallel over (b,ic),
  ! id and ir sequential inside: race-free, no reduction needed.
  !$omp parallel do collapse(2) private(base_ic, id)
  do b_i = 0, B-1
    do ic = 0, N-1
      base_ic = (b_i*N+ic)*D
      do id = 0, D-1
        block
          real(8) :: g
          integer :: irr
          g = 0.0d0
          do irr = 0, N-1
            g = g + A((b_i*N+irr)*N+ic+1) * dO((b_i*N+irr)*D+id+1)
          end do
          GV(base_ic+id+1) = g
        end block
      end do
    end do
  end do
  !$omp end parallel do

  ! GQ, GK: parallel over (b,ir). GK is accumulated across different ir
  ! values writing the same ic-indexed slot -- needs an explicit array
  ! reduction, same as backward_bench.c's OpenMP reduction on GK.
  !$omp parallel do collapse(2) private(base_ir) reduction(+:GK)
  do b_i = 0, B-1
    do ir = 0, N-1
      base_ir = (b_i*N+ir)*D
      block
        real(8) :: ga_row(N), corr, gs
        integer :: icc, jj
        do icc = 0, N-1
          ga_row(icc+1) = 0.0d0
          do jj = 0, D-1
            ga_row(icc+1) = ga_row(icc+1) + dO(base_ir+jj+1) * V((b_i*N+icc)*D+jj+1)
          end do
        end do
        corr = 0.0d0
        do icc = 0, N-1
          corr = corr + A((b_i*N+ir)*N+icc+1) * ga_row(icc+1)
        end do
        do jj = 0, D-1
          GQ(base_ir+jj+1) = 0.0d0
        end do
        do icc = 0, N-1
          gs = scale * A((b_i*N+ir)*N+icc+1) * (ga_row(icc+1) - corr)
          do jj = 0, D-1
            GQ(base_ir+jj+1) = GQ(base_ir+jj+1) + gs * K((b_i*N+icc)*D+jj+1)
            GK((b_i*N+icc)*D+jj+1) = GK((b_i*N+icc)*D+jj+1) + gs * Q(base_ir+jj+1)
          end do
        end do
      end block
    end do
  end do
  !$omp end parallel do

  t1 = omp_get_wtime()

  print '(A,I0,A,I0,A,I0,A,I0,A,F10.6)', &
    "backward,Fortran,", B, ",", N, ",", D, ",", THREADS, ",", t1-t0

  if (do_check) then
    allocate(GV_ref(NT*D), GQ_ref(NT*D), GK_ref(NT*D))
    GV_ref = 0.0d0; GQ_ref = 0.0d0; GK_ref = 0.0d0
    call backward_serial(Q, K, V, A, dO, GV_ref, GQ_ref, GK_ref, scale, B, N, D)
    print '(A,ES12.3)', "check: GV max diff = ", maxval(abs(GV - GV_ref))
    print '(A,ES12.3)', "check: GQ max diff = ", maxval(abs(GQ - GQ_ref))
    print '(A,ES12.3)', "check: GK max diff = ", maxval(abs(GK - GK_ref))
    deallocate(GV_ref, GQ_ref, GK_ref)
  end if

  deallocate(Q, K, V, dO, A, GV, GQ, GK)

contains

  subroutine forward_serial(Q, K, V, A, scale, B, N, D)
    real(8), intent(in) :: Q(*), K(*), V(*), scale
    real(8), intent(out) :: A(*)
    integer, intent(in) :: B, N, D
    integer :: b_i, ir, ic, jj, base_ir, base_ic
    real(8) :: arow_l(N), s, m, sm
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

end program backward_bench
