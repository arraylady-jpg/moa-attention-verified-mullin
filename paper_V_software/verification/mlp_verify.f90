program mlp_verify
  implicit none
  integer, parameter :: B = 2, N = 4, D = 8, DFF = 16, NT = N*B
  real(8) :: X(D*NT), Wg(DFF*D), Wu(DFF*D), Wd(D*DFF), GF(D*NT)
  real(8) :: U(DFF*NT), Fout(D*NT)
  real(8) :: GX(D*NT), GWg(DFF*D), GWu(DFF*D), GWd(D*DFF)
  real(8) :: Vrow(DFF), Hrow(DFF), GHrow(DFF), GUrow(DFF), GVrow(DFF)
  real(8) :: u_val, v_val, f_val, gh, gx_val
  integer :: t, d_i, f_i, xbase, ubase, wgbase, wubase, wdbase

  ! Flat index conventions (all 0-based math, matching numpy row-major layout):
  !   X, GF, GX, F  : (NT, D)   base = t*D,   +d
  !   U             : (NT, DFF) base = t*DFF, +f
  !   Wg, Wu        : (D, DFF)  base = d*DFF, +f   (python shape (D,DFF))
  !   Wd            : (DFF, D)  base = f*D,   +d   (python shape (DFF,D))

  call load_flat("data/X.bin", X, D*NT)
  call load_flat("data/Wg.bin", Wg, D*DFF)
  call load_flat("data/Wu.bin", Wu, D*DFF)
  call load_flat("data/Wd.bin", Wd, DFF*D)
  call load_flat("data/GF.bin", GF, D*NT)

  ! ---- Forward DNF (Section 6): U = X x Wg, V (not retained), H = silu(U)*V, F = H x Wd ----
  !$omp parallel do private(xbase, ubase, d_i, f_i, u_val, v_val, f_val, wgbase, wdbase, Vrow, Hrow)
  do t = 0, NT-1
    xbase = t*D
    ubase = t*DFF
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

  call save_flat("data/f_F.bin", Fout, D*NT)

  ! ---- Backward DNF (Sections 6-7): V,H recomputed on the fly from retained U ----
  !      Every "transposed" access below (Wd^T, Wg^T, Wu^T, X^T, H^T) is a reordered-index
  !      read from the original array -- no transpose is ever materialized (Section 8 lemma).
  GX = 0.0d0
  GWg = 0.0d0
  GWu = 0.0d0
  GWd = 0.0d0

  !$omp parallel do private(xbase, ubase, d_i, f_i, v_val, gh, gx_val, wgbase, wdbase, &
  !$omp                     Vrow, Hrow, GHrow, GUrow, GVrow) reduction(+:GWg, GWu, GWd)
  do t = 0, NT-1
    xbase = t*D
    ubase = t*DFF
    do f_i = 0, DFF-1
      v_val = 0.0d0
      do d_i = 0, D-1
        wgbase = d_i*DFF
        v_val = v_val + X(xbase+d_i+1) * Wu(wgbase+f_i+1)
      end do
      Vrow(f_i+1) = v_val
      Hrow(f_i+1) = silu(U(ubase+f_i+1)) * v_val
    end do

    ! G_Wd = H^T x G_F -- H accessed by (f,d) reordering, never transposed in storage
    do f_i = 0, DFF-1
      wdbase = f_i*D
      do d_i = 0, D-1
        GWd(wdbase+d_i+1) = GWd(wdbase+d_i+1) + Hrow(f_i+1) * GF(xbase+d_i+1)
      end do
    end do

    ! G_H = G_F x Wd^T -- Wd accessed by (f,d) reordering, never transposed in storage
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

    ! G_Wg = X^T x G_U, G_Wu = X^T x G_V -- X accessed by (d,t) reordering, never transposed
    do d_i = 0, D-1
      wgbase = d_i*DFF
      do f_i = 0, DFF-1
        GWg(wgbase+f_i+1) = GWg(wgbase+f_i+1) + X(xbase+d_i+1) * GUrow(f_i+1)
        GWu(wgbase+f_i+1) = GWu(wgbase+f_i+1) + X(xbase+d_i+1) * GVrow(f_i+1)
      end do
    end do

    ! G_X = G_U x Wg^T + G_V x Wu^T -- Wg, Wu accessed by (d,f) reordering, never transposed
    do d_i = 0, D-1
      wgbase = d_i*DFF
      gx_val = 0.0d0
      do f_i = 0, DFF-1
        gx_val = gx_val + GUrow(f_i+1) * Wg(wgbase+f_i+1) + GVrow(f_i+1) * Wu(wgbase+f_i+1)
      end do
      GX(xbase+d_i+1) = gx_val
    end do
  end do
  !$omp end parallel do

  call save_flat("data/f_GX_ffn.bin", GX, D*NT)
  call save_flat("data/f_GWg.bin", GWg, D*DFF)
  call save_flat("data/f_GWu.bin", GWu, D*DFF)
  call save_flat("data/f_GWd.bin", GWd, DFF*D)

  print *, "Gated MLP forward+backward complete. F(1..4) = ", Fout(1), Fout(2), Fout(3), Fout(4)

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

end program mlp_verify
