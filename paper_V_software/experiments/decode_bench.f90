! decode_bench.f90
! Fortran90/OpenMP counterpart to moa_decode_bench.c's moa_decode_omp
! (Paper III decode kernel: single query against n cached keys/values).
! Same three-pass structure (scores+max, softmax, weighted sum).
!
! DEVIATION FROM SOURCE: moa_decode_bench.c uses real(4) (float); this
! file uses real(8) (double), for consistency with every other Fortran
! bench file in this project and for exact --check tolerance.
!
! Usage: ./decode_bench_f90 max_threads n_start n_max dk dv repeats [--check]
program decode_bench
  use omp_lib
  implicit none
  integer :: max_threads, n_start, n_max, dk, dv, repeats
  integer :: nt, n, r
  real(8), allocatable :: q(:), K(:), V(:), out(:), out_ref(:)
  real(8) :: scale, t0, elapsed, traffic_mb, gbps
  character(len=32) :: arg
  logical :: do_check
  integer :: csv_unit

  if (command_argument_count() < 6) then
    print *, "usage: decode_bench_f90 max_threads n_start n_max dk dv repeats [--check]"
    stop 1
  end if
  call get_command_argument(1, arg); read(arg,*) max_threads
  call get_command_argument(2, arg); read(arg,*) n_start
  call get_command_argument(3, arg); read(arg,*) n_max
  call get_command_argument(4, arg); read(arg,*) dk
  call get_command_argument(5, arg); read(arg,*) dv
  call get_command_argument(6, arg); read(arg,*) repeats
  do_check = .false.
  if (command_argument_count() >= 7) then
    call get_command_argument(7, arg)
    if (trim(arg) == "--check") do_check = .true.
  end if

  print *, "=== MoA Decode Fortran90/OpenMP Benchmark ==="

  open(newunit=csv_unit, file="timings_cpu_f90.csv", status="replace")
  write(csv_unit, '(A)') "threads,n,dk,dv,time_s,gbps,traffic_mb"

  nt = 1
  do while (nt <= max_threads)
    n = n_start
    do while (n <= n_max)
      allocate(q(dk), K(n*dk), V(n*dv), out(dv))
      call fill_rand(q, dk, 1)
      call fill_rand(K, n*dk, 2)
      call fill_rand(V, n*dv, 3)
      scale = 1.0d0 / sqrt(dble(dk))

      call omp_set_num_threads(nt)

      if (do_check) then
        allocate(out_ref(dv))
        call decode_serial(q, K, V, out_ref, scale, n, dk, dv)
        call decode_omp(q, K, V, out, scale, n, dk, dv)
        print '(A,I0,A,I0,A,ES12.3)', "check: threads=", nt, " n=", n, &
              " max diff = ", maxval(abs(out - out_ref))
        deallocate(out_ref)
      end if

      do r = 1, 2
        call decode_omp(q, K, V, out, scale, n, dk, dv)
      end do

      t0 = omp_get_wtime()
      do r = 1, repeats
        call decode_omp(q, K, V, out, scale, n, dk, dv)
      end do
      elapsed = (omp_get_wtime() - t0) / dble(repeats)

      traffic_mb = dble(dk + n*dk + n*dv + dv) * 8.0d0 / 1.0d6
      gbps = (traffic_mb / 1.0d3) / elapsed

      print '(A,I0,A,I0,A,F12.6,A,F8.3)', &
        "threads=", nt, " n=", n, " time_s=", elapsed, " GB/s=", gbps
      write(csv_unit, '(I0,A,I0,A,I0,A,I0,A,F12.8,A,F8.4,A,F8.4)') &
        nt, ",", n, ",", dk, ",", dv, ",", elapsed, ",", gbps, ",", traffic_mb

      deallocate(q, K, V, out)
      n = n * 2
    end do
    nt = nt * 2
  end do

  close(csv_unit)
  print *, "Results saved to timings_cpu_f90.csv"

contains

  subroutine decode_omp(q, K, V, out, scale, n, dk, dv)
    real(8), intent(in) :: q(*), K(*), V(*), scale
    real(8), intent(out) :: out(*)
    integer, intent(in) :: n, dk, dv
    real(8), allocatable :: s(:), e(:)
    real(8) :: m, Z, inv_Z, dot, acc
    integer :: l, j, d
    allocate(s(n), e(n))

    m = -1.0d300
    !$omp parallel do reduction(max:m) private(dot, j)
    do l = 0, n-1
      dot = 0.0d0
      do j = 0, dk-1
        dot = dot + q(j+1) * K(l*dk+j+1)
      end do
      s(l+1) = scale * dot
      if (s(l+1) > m) m = s(l+1)
    end do
    !$omp end parallel do

    Z = 0.0d0
    !$omp parallel do reduction(+:Z)
    do l = 0, n-1
      e(l+1) = exp(s(l+1) - m)
      Z = Z + e(l+1)
    end do
    !$omp end parallel do
    inv_Z = 1.0d0 / Z

    !$omp parallel do private(acc, l)
    do d = 0, dv-1
      acc = 0.0d0
      do l = 0, n-1
        acc = acc + (e(l+1) * inv_Z) * V(l*dv+d+1)
      end do
      out(d+1) = acc
    end do
    !$omp end parallel do

    deallocate(s, e)
  end subroutine decode_omp

  subroutine decode_serial(q, K, V, out, scale, n, dk, dv)
    real(8), intent(in) :: q(*), K(*), V(*), scale
    real(8), intent(out) :: out(*)
    integer, intent(in) :: n, dk, dv
    real(8), allocatable :: s(:), e(:)
    real(8) :: m, Z, inv_Z, dot, acc
    integer :: l, j, d
    allocate(s(n), e(n))
    m = -1.0d300
    do l = 0, n-1
      dot = 0.0d0
      do j = 0, dk-1
        dot = dot + q(j+1) * K(l*dk+j+1)
      end do
      s(l+1) = scale * dot
      if (s(l+1) > m) m = s(l+1)
    end do
    Z = 0.0d0
    do l = 0, n-1
      e(l+1) = exp(s(l+1) - m)
      Z = Z + e(l+1)
    end do
    inv_Z = 1.0d0 / Z
    do d = 0, dv-1
      acc = 0.0d0
      do l = 0, n-1
        acc = acc + (e(l+1) * inv_Z) * V(l*dv+d+1)
      end do
      out(d+1) = acc
    end do
    deallocate(s, e)
  end subroutine decode_serial

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

end program decode_bench
