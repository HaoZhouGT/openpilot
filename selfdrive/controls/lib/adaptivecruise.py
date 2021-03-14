import math
import numpy as np
from common.numpy_fast import clip, interp
from selfdrive.config import Conversions as CV


# lookup tables VS speed to determine min and max accels in cruise
_A_CRUISE_MIN_V  = [-1.0, -.8, -.67, -.5, -.30]
_A_CRUISE_MIN_BP = [   0., 5.,  10., 20.,  40.]

# need fast accel at very low speed for stop and go
_A_CRUISE_MAX_V  = [1., 1., .8, .5, .30]
_A_CRUISE_MAX_BP = [0.,  5., 10., 20., 40.]

def calc_cruise_accel_limits(v_ego):
  a_cruise_min = interp(v_ego, _A_CRUISE_MIN_BP, _A_CRUISE_MIN_V)
  a_cruise_max = interp(v_ego, _A_CRUISE_MAX_BP, _A_CRUISE_MAX_V)
  return np.vstack([a_cruise_min, a_cruise_max])

_A_TOTAL_MAX_V = [1.5, 1.9, 3.2]
_A_TOTAL_MAX_BP = [0., 20., 40.]

def limit_accel_in_turns(v_ego, angle_steers, a_target, a_pcm, CP):
  #*** this function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  # this should avoid accelerating when losing the target in turns
  # deg_to_rad = np.pi / 180.  # from can reading to rad

  a_total_max = interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego**2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  a_x_allowed = math.sqrt(max(a_total_max**2 - a_y**2, 0.))

  a_target[1] = min(a_target[1], a_x_allowed)
  a_pcm = min(a_pcm, a_x_allowed)
  return a_target, a_pcm

def process_a_lead(a_lead):
  # soft threshold of 0.5m/s^2 applied to a_lead to reject noise, also not considered positive a_lead
  a_lead_threshold = 0.5
  a_lead = min(a_lead + a_lead_threshold, 0)
  return a_lead


IDM_a_max = 1.5 # comfortable max acceleration
IDM_b_max = 2.0 # comfortable max deceleration
s0 = 4.0 # minimum distance
accel_expo = 4
T = 1.5 # desired headway

def IDM(vCruise, d_lead, vEgo, v_lead, a_lead):
  d_lead = max(d_lead, 0.1) # do not have zero values
	v_rel = vEgo - v_lead

	s = s0 + vEgo * T + vEgo*v_rel/(2*np.sqrt(IDM_a_max*IDM_b_max))
	aTarget = IDM_a_max*(1-(vEgo/vCruise)**accel_expo-(s/d_lead)**2)
	return aTarget


MAX_SPEED_POSSIBLE = 55.

def compute_IDM_accel(v_cruise_setpoint, v_ego, angle_steers, l1, l2, CP):
  v_cruise_setpoint = min(v_cruise_setpoint, 45.0)
  v_cruise_setpoint = max(0.1, v_cruise_setpoint)
  # drive limits
  # TODO: Make lims function of speed (more aggressive at low speed).
  a_lim = [-3., 1.5]

  #*** set accel limits as cruise accel/decel limits ***
  a_limits= calc_cruise_accel_limits(v_ego)
  # Always 1 for now.
  a_pcm = 1

  #*** limit max accel in sharp turns
  a_limits, a_pcm = limit_accel_in_turns(v_ego, angle_steers, a_limits, a_pcm, CP)
  jerk_factor = 0.
  aTarget = float(0.0) # default value if no lead is detected

  if l1 is not None and l1.status:
    #*** process noisy a_lead signal from radar processing ***
    a_lead_p = process_a_lead(l1.aLeadK)

    aTarget = IDM(v_cruise_setpoint, l1.dRel, v_ego, l1.vLead, a_lead_p)

    if l2 is not None and l2.status:
      #*** process noisy a_lead signal from radar processing ***
      a_lead_p2 = process_a_lead(l2.aLeadK)
      aTarget2 = IDM(v_cruise_setpoint, l2.dRel, v_ego, l2.vLead, a_lead_p2)

      # listen to lead that makes the acceleration smaller
      if aTarget2 < aTarget:
        l1 = l2
        aTarget = aTarget2

    # l1 is the main lead now
    # we can now limit a_target to a_lim
    aTarget = clip(aTarget, a_limits[0], a_limits[1])

  return aTarget



class AdaptiveCruise(object):
  def __init__(self):
    self.last_cal = 0.
    self.l1, self.l2 = None, None
    self.dead = True
    self.aTarget = float(0.0) # default
  def update(self, cur_time, v_cruise_setpoint, v_ego, angle_steers, CP, lead1, lead2): # the update can be called, thus the arguments here can be accessed
    # TODO: no longer has anything to do with calibration
    self.last_cal = cur_time
    self.dead = False
    if cur_time - self.last_cal > 0.5:
      self.dead = True
    self.aTarget = compute_IDM_accel(v_cruise_setpoint, v_ego, angle_steers, lead1, lead2, CP)