! forward_bench.f90
! Fortran90/OpenMP counterpart to forward_bench.c (Paper I forward pass,
! arXiv:2606.07713v1 eqs. 17, 20, 21). Same math, same flat-array
! indexing convention as rmsnorm_bench.f90/mlp_bench.f90 (Paper IV).
!
! Usage: ./forward_bench_f90 B N D THREADS [--check]
! Prints: forward,Fortran,B,N,D,threads,seconds
program forward_bench
  use omp_lib
  implicit none
  integer :: B, N, D, THREADS, NT
  real(8), allocatable :: Q(:), K(:), V(:), A(:), Out(:)
  real(8), allocatable :: A_ref(:), Out_ref(:)
  real(8) :: t0, t1, scale
  integer :: b_i, ir, ic, id, base_ir, base_ic
  character(len=32) :: arg
  logical :: do_check

  if (command_argument_count() < 4) then
    print *, "usage: forward_bench_f90 B N D THREADS [--check]"
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

  allocate(Q(NT*D), K(NT*D), V(NT*D), A(NT*N), Out(NT*D))
  call fill_rand(Q, NT*D, 1)
  call fill_rand(K, NT*D, 2)
  call fill_rand(V, NT*D, 3)

  t0 = omp_get_wtime()

  ! Flat indexing: QKV(b,i,d) = ((b*N+i)*D)+d+1 (0-based b,i,d -> 1-based Fortran)
  ! AA(b,i,j)   = ((b*N+i)*N)+j+1
  !$omp parallel do collapse(2) private(base_ir, base_ic, id, ic)
  do b_i = 0, B-1
    do ir = 0, N-1
      block
        real(8) :: arow_l(N)
        real(8) :: s, m, sm, o
        integer :: jj
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
            base_ic = (b_i*N+ic)*D
            o = o + arow_l(ic+1) * V(base_ic+id+1)
          end do
          Out(base_ir+id+1) = o
        end do
      end block
    end do
  end do
  !$omp end parallel do

  t1 = omp_get_wtime()

  print '(A,I0,A,I0,A,I0,A,I0,A,F10.6)', &
    "forward,Fortran,", B, ",", N, ",", D, ",", THREADS, ",", t1-t0

  if (do_check) then
    allocate(A_ref(NT*N), Out_ref(NT*D))
    call forward_serial(Q, K, V, A_ref, Out_ref, scale, B, N, D)
    print '(A,ES12.3)', "check: A   max diff = ", maxval(abs(A - A_ref))
    print '(A,ES12.3)', "check: Out max diff = ", maxval(abs(Out - Out_ref))
    deallocate(A_ref, Out_ref)
  end if

  deallocate(Q, K, V, A, Out)

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
            base_ic = (b_i*N+ic)*D
            o = o + arow_l(ic+1) * V(base_ic+id+1)
          end do
          Out(base_ir+id+1) = o
        end do
      end do
    end do
  end subroutine forward_serial

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

end program forward_bench
