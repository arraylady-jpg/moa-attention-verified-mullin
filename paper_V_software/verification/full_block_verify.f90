! full_block_verify.f90
! Fortran90 port of full_block_verify.c: complete transformer block
! (Norm -> QKV projection -> Attention -> residual -> Norm -> FFN ->
! residual), forward + backward, using the same fb_*.bin data files
! written by reference_full_block.py (PyTorch reference). Same flat-
! array indexing convention as rmsnorm_verify.f90/mlp_verify.f90.
!
! Compile: gfortran -O2 -o full_block_verify_f90 full_block_verify.f90
! Run from the same directory as data/fb_*.bin (i.e. after running
! reference_full_block.py).
program full_block_verify
  implicit none
  integer, parameter :: B = 2, N = 4, D = 8, DFF = 16, NT = B*N
  real(8), parameter :: EPS = 1.0d-6

  real(8) :: X(NT*D), gamma1(D), gamma2(D)
  real(8) :: Wq(D*D), Wk(D*D), Wv(D*D), Wg(D*DFF), Wu(D*DFF), Wd(DFF*D)
  real(8) :: GY(NT*D)
  real(8) :: Z1(NT*D), rn1(NT), Q(NT*D), K(NT*D), V(NT*D), A(B*N*N)
  real(8) :: AttnOut(NT*D), R1(NT*D), Z2(NT*D), rn2(NT)
  real(8) :: U(NT*DFF), Vg(NT*DFF), H(NT*DFF), Y(NT*D)
  real(8) :: GR1(NT*D), GZ2(NT*D), GWg(D*DFF), GWu(D*DFF), GWd(DFF*D), Ggamma2(D)
  real(8) :: GX(NT*D), GQ(NT*D), GK(NT*D), GV(NT*D)
  real(8) :: GZ1(NT*D), GWq(D*D), GWk(D*D), GWv(D*D), Ggamma1(D)

  real(8) :: scale, sumsq, xval, m, s, o, dot, gvv, corr, gs, gh, gz, correction
  real(8) :: x_hat_d, x_hat_e, fo, u_val, v_val
  real(8) :: arow(N), ga_row(N), GHrow(DFF), GUrow(DFF), GVrow(DFF)
  integer :: t, d_i, d2, b_i, ir, ic, j, f_i, e_i, i, base_t, base_ic, base_ir

  scale = 1.0d0 / sqrt(dble(D))

  call load_flat("data/fb_X.bin", X, NT*D)
  call load_flat("data/fb_gamma1.bin", gamma1, D)
  call load_flat("data/fb_gamma2.bin", gamma2, D)
  call load_flat("data/fb_Wq.bin", Wq, D*D)
  call load_flat("data/fb_Wk.bin", Wk, D*D)
  call load_flat("data/fb_Wv.bin", Wv, D*D)
  call load_flat("data/fb_Wg.bin", Wg, D*DFF)
  call load_flat("data/fb_Wu.bin", Wu, D*DFF)
  call load_flat("data/fb_Wd.bin", Wd, DFF*D)
  call load_flat("data/fb_GY.bin", GY, NT*D)

  ! ================= FORWARD =================

  ! ---- Norm 1 ----
  do t = 0, NT-1
    base_t = t*D
    sumsq = 0.0d0
    do d_i = 0, D-1
      xval = X(base_t+d_i+1)
      sumsq = sumsq + xval*xval
    end do
    rn1(t+1) = sqrt(sumsq/dble(D) + EPS)
    do d_i = 0, D-1
      Z1(base_t+d_i+1) = gamma1(d_i+1) * (X(base_t+d_i+1) / rn1(t+1))
    end do
  end do

  ! ---- QKV projection: Q=Z1@Wq, K=Z1@Wk, V=Z1@Wv ----
  do t = 0, NT-1
    base_t = t*D
    do d2 = 0, D-1
      block
        real(8) :: qv, kv, vv
        integer :: dd
        qv = 0.0d0; kv = 0.0d0; vv = 0.0d0
        do dd = 0, D-1
          qv = qv + Z1(base_t+dd+1) * Wq(dd*D+d2+1)
          kv = kv + Z1(base_t+dd+1) * Wk(dd*D+d2+1)
          vv = vv + Z1(base_t+dd+1) * Wv(dd*D+d2+1)
        end do
        Q(base_t+d2+1) = qv; K(base_t+d2+1) = kv; V(base_t+d2+1) = vv
      end block
    end do
  end do

  ! ---- Attention forward (DNF), per batch ----
  do b_i = 0, B-1
    do ir = 0, N-1
      m = -1.0d300
      do ic = 0, N-1
        dot = 0.0d0
        do j = 0, D-1
          dot = dot + Q((b_i*N+ir)*D+j+1) * K((b_i*N+ic)*D+j+1)
        end do
        arow(ic+1) = scale * dot
        if (arow(ic+1) > m) m = arow(ic+1)
      end do
      s = 0.0d0
      do ic = 0, N-1
        arow(ic+1) = exp(arow(ic+1) - m)
        s = s + arow(ic+1)
      end do
      do ic = 0, N-1
        arow(ic+1) = arow(ic+1) / s
        A((b_i*N+ir)*N+ic+1) = arow(ic+1)
      end do
      do d_i = 0, D-1
        o = 0.0d0
        do ic = 0, N-1
          o = o + arow(ic+1) * V((b_i*N+ic)*D+d_i+1)
        end do
        AttnOut((b_i*N+ir)*D+d_i+1) = o
      end do
    end do
  end do

  ! ---- Residual 1 ----
  do i = 1, NT*D
    R1(i) = X(i) + AttnOut(i)
  end do

  ! ---- Norm 2 ----
  do t = 0, NT-1
    base_t = t*D
    sumsq = 0.0d0
    do d_i = 0, D-1
      xval = R1(base_t+d_i+1)
      sumsq = sumsq + xval*xval
    end do
    rn2(t+1) = sqrt(sumsq/dble(D) + EPS)
    do d_i = 0, D-1
      Z2(base_t+d_i+1) = gamma2(d_i+1) * (R1(base_t+d_i+1) / rn2(t+1))
    end do
  end do

  ! ---- Gated MLP forward ----
  do t = 0, NT-1
    base_t = t*D
    do f_i = 0, DFF-1
      u_val = 0.0d0; v_val = 0.0d0
      do d_i = 0, D-1
        u_val = u_val + Z2(base_t+d_i+1) * Wg(d_i*DFF+f_i+1)
        v_val = v_val + Z2(base_t+d_i+1) * Wu(d_i*DFF+f_i+1)
      end do
      U(t*DFF+f_i+1) = u_val
      Vg(t*DFF+f_i+1) = v_val
      H(t*DFF+f_i+1) = silu(u_val) * v_val
    end do
  end do
  do t = 0, NT-1
    base_t = t*D
    do d_i = 0, D-1
      fo = 0.0d0
      do f_i = 0, DFF-1
        fo = fo + H(t*DFF+f_i+1) * Wd(f_i*D+d_i+1)
      end do
      Y(base_t+d_i+1) = R1(base_t+d_i+1) + fo
    end do
  end do

  call save_flat("data/fb_f_Y.bin", Y, NT*D)

  ! ================= BACKWARD =================
  GR1 = 0.0d0; GZ2 = 0.0d0; GWg = 0.0d0; GWu = 0.0d0; GWd = 0.0d0; Ggamma2 = 0.0d0

  do i = 1, NT*D
    GR1(i) = GR1(i) + GY(i)
  end do

  ! FFN backward, feeding G_Z2
  do t = 0, NT-1
    base_t = t*D
    do f_i = 0, DFF-1
      gh = 0.0d0
      do d_i = 0, D-1
        gh = gh + GY(base_t+d_i+1) * Wd(f_i*D+d_i+1)
      end do
      GHrow(f_i+1) = gh
      GVrow(f_i+1) = gh * silu(U(t*DFF+f_i+1))
      GUrow(f_i+1) = gh * Vg(t*DFF+f_i+1) * silu_grad(U(t*DFF+f_i+1))
    end do
    do f_i = 0, DFF-1
      do d_i = 0, D-1
        GWd(f_i*D+d_i+1) = GWd(f_i*D+d_i+1) + H(t*DFF+f_i+1) * GY(base_t+d_i+1)
      end do
    end do
    do d_i = 0, D-1
      do f_i = 0, DFF-1
        GWg(d_i*DFF+f_i+1) = GWg(d_i*DFF+f_i+1) + Z2(base_t+d_i+1) * GUrow(f_i+1)
        GWu(d_i*DFF+f_i+1) = GWu(d_i*DFF+f_i+1) + Z2(base_t+d_i+1) * GVrow(f_i+1)
      end do
    end do
    do d_i = 0, D-1
      gz = 0.0d0
      do f_i = 0, DFF-1
        gz = gz + GUrow(f_i+1)*Wg(d_i*DFF+f_i+1) + GVrow(f_i+1)*Wu(d_i*DFF+f_i+1)
      end do
      GZ2(base_t+d_i+1) = gz
    end do
  end do

  ! Norm 2 backward, accumulating into GR1
  do t = 0, NT-1
    base_t = t*D
    correction = 0.0d0
    do d_i = 0, D-1
      x_hat_d = R1(base_t+d_i+1) / rn2(t+1)
      correction = correction + GZ2(base_t+d_i+1) * gamma2(d_i+1) * x_hat_d
    end do
    do e_i = 0, D-1
      x_hat_e = R1(base_t+e_i+1) / rn2(t+1)
      GR1(base_t+e_i+1) = GR1(base_t+e_i+1) + &
        (GZ2(base_t+e_i+1)*gamma2(e_i+1) - (x_hat_e/dble(D))*correction) / rn2(t+1)
      Ggamma2(e_i+1) = Ggamma2(e_i+1) + GZ2(base_t+e_i+1) * x_hat_e
    end do
  end do

  ! G_AttnOut = G_R1 (residual 1 also flows to G_X directly)
  GX = 0.0d0
  do i = 1, NT*D
    GX(i) = GX(i) + GR1(i)
  end do

  ! Attention backward, feeding G_Q, G_K, G_V
  GQ = 0.0d0; GK = 0.0d0; GV = 0.0d0
  do b_i = 0, B-1
    do ic = 0, N-1
      do d_i = 0, D-1
        gvv = 0.0d0
        do ir = 0, N-1
          gvv = gvv + A((b_i*N+ir)*N+ic+1) * GR1((b_i*N+ir)*D+d_i+1)
        end do
        GV((b_i*N+ic)*D+d_i+1) = gvv
      end do
    end do
    do ir = 0, N-1
      base_ir = (b_i*N+ir)*D
      do ic = 0, N-1
        block
          real(8) :: gg
          integer :: dd
          gg = 0.0d0
          do dd = 0, D-1
            gg = gg + GR1(base_ir+dd+1) * V((b_i*N+ic)*D+dd+1)
          end do
          ga_row(ic+1) = gg
        end block
      end do
      corr = 0.0d0
      do ic = 0, N-1
        corr = corr + A((b_i*N+ir)*N+ic+1) * ga_row(ic+1)
      end do
      do ic = 0, N-1
        gs = scale * A((b_i*N+ir)*N+ic+1) * (ga_row(ic+1) - corr)
        base_ic = (b_i*N+ic)*D
        do j = 0, D-1
          GQ(base_ir+j+1) = GQ(base_ir+j+1) + gs * K(base_ic+j+1)
          GK(base_ic+j+1) = GK(base_ic+j+1) + gs * Q(base_ir+j+1)
        end do
      end do
    end do
  end do

  ! QKV projection backward, feeding G_Z1
  GZ1 = 0.0d0; GWq = 0.0d0; GWk = 0.0d0; GWv = 0.0d0
  do t = 0, NT-1
    base_t = t*D
    do d_i = 0, D-1
      do d2 = 0, D-1
        GWq(d_i*D+d2+1) = GWq(d_i*D+d2+1) + Z1(base_t+d_i+1) * GQ(base_t+d2+1)
        GWk(d_i*D+d2+1) = GWk(d_i*D+d2+1) + Z1(base_t+d_i+1) * GK(base_t+d2+1)
        GWv(d_i*D+d2+1) = GWv(d_i*D+d2+1) + Z1(base_t+d_i+1) * GV(base_t+d2+1)
      end do
    end do
    do d_i = 0, D-1
      block
        real(8) :: gg
        integer :: dd2
        gg = 0.0d0
        do dd2 = 0, D-1
          gg = gg + GQ(base_t+dd2+1)*Wq(d_i*D+dd2+1) + GK(base_t+dd2+1)*Wk(d_i*D+dd2+1) &
                  + GV(base_t+dd2+1)*Wv(d_i*D+dd2+1)
        end do
        GZ1(base_t+d_i+1) = gg
      end block
    end do
  end do

  ! Norm 1 backward, accumulating into GX
  Ggamma1 = 0.0d0
  do t = 0, NT-1
    base_t = t*D
    correction = 0.0d0
    do d_i = 0, D-1
      x_hat_d = X(base_t+d_i+1) / rn1(t+1)
      correction = correction + GZ1(base_t+d_i+1) * gamma1(d_i+1) * x_hat_d
    end do
    do e_i = 0, D-1
      x_hat_e = X(base_t+e_i+1) / rn1(t+1)
      GX(base_t+e_i+1) = GX(base_t+e_i+1) + &
        (GZ1(base_t+e_i+1)*gamma1(e_i+1) - (x_hat_e/dble(D))*correction) / rn1(t+1)
      Ggamma1(e_i+1) = Ggamma1(e_i+1) + GZ1(base_t+e_i+1) * x_hat_e
    end do
  end do

  call save_flat("data/fb_f_GX.bin", GX, NT*D)
  call save_flat("data/fb_f_Ggamma1.bin", Ggamma1, D)
  call save_flat("data/fb_f_Ggamma2.bin", Ggamma2, D)
  call save_flat("data/fb_f_GWq.bin", GWq, D*D)
  call save_flat("data/fb_f_GWk.bin", GWk, D*D)
  call save_flat("data/fb_f_GWv.bin", GWv, D*D)
  call save_flat("data/fb_f_GWg.bin", GWg, D*DFF)
  call save_flat("data/fb_f_GWu.bin", GWu, D*DFF)
  call save_flat("data/fb_f_GWd.bin", GWd, DFF*D)

  print '(A,4F10.6)', "Full block forward+backward complete. Y(1..4) = ", Y(1), Y(2), Y(3), Y(4)
  print '(A,4F10.6)', "GX(1..4) = ", GX(1), GX(2), GX(3), GX(4)

contains

  real(8) function silu(u)
    real(8), intent(in) :: u
    silu = u / (1.0d0 + exp(-u))
  end function silu

  real(8) function silu_grad(u)
    real(8), intent(in) :: u
    real(8) :: sg
    sg = 1.0d0 / (1.0d0 + exp(-u))
    silu_grad = sg * (1.0d0 + u * (1.0d0 - sg))
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

end program full_block_verify
