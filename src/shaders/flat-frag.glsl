#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;

in vec2 fs_Pos;
out vec4 out_Col;

#define EPSILON 0.0001
#define NORMAL_EPSILON 0.01
#define MAX_DIST 1000.0
#define MAX_MARCHING_STEPS 500
#define EYE_RADIUS 5.0
#define PUPIL_RADIUS 4.0
#define M_PI 3.1415926535897932384626433832795

mat4 rotationMatrix(vec3 axis, float angle)
{
    axis = normalize(axis);
    float s = -sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
}

float random1(vec2 p, vec2 seed) {
  return fract(sin(dot(p + seed, vec2(127.1, 311.7))) * 43758.5453);
}

vec2 random2(vec2 p, vec2 seed) {
  return fract(sin(vec2(dot(p + seed, vec2(311.7, 127.1)), dot(p + seed, vec2(269.5, 183.3)))) * 85734.3545);
}

vec2 randvec1(vec2 n, vec2 seed) {
    float x = sin(dot(n + seed, vec2(131.32, 964.31)));
    float y = sin(dot(n + seed, vec2(139.345, 132.89)));
    vec2 v = fract(329.779f * vec2(x, y));
    return vec2(2.0 * v.x - 1.0, 2.0 * v.y - 1.0);
}

vec2 randvec2(vec2 n, vec2 seed) {
    float x = sin(dot(n + seed, vec2(113.2, 634.11)));
    float y = sin(dot(n + seed, vec2(109.5, 242.8)));
    return fract(3242.177f * vec2(x, y));
}

vec2 randvec3(vec2 n, vec2 seed) {
  float x = sin(dot(n + seed, vec2(14.92, 64.42)));
  float y = sin(dot(n + seed, vec2(48.12, 32.42)));
  return fract(334.963f * vec2(x, y));
}

float fireNoise(vec3 uv, float res)
{
	const vec3 s = vec3(1.0, 100.0, 1000.0);
	uv *= res;
	vec3 uv0 = floor(mod(uv, res)) * s;
	vec3 uv1 = floor(mod(uv + vec3(1.0), res)) * s;
	vec3 f = fract(uv); f = f * f * (3.0 - 2.0 * f);

	vec4 v = vec4(uv0.x + uv0.y + uv0.z, uv1.x + uv0.y + uv0.z, uv0.x + uv1.y + uv0.z, uv1.x + uv1.y + uv0.z);
	vec4 r = fract(sin(v * 0.1) * 1000.0);
	float r0 = mix(mix(r.x, r.y, f.x), mix(r.z, r.w, f.x), f.y);
	r = fract(sin((v + uv1.z - uv0.z) * 0.1) * 1000.0);
	float r1 = mix(mix(r.x, r.y, f.x), mix(r.z, r.w, f.x), f.y);
	return mix(r0, r1, f.z) * 2.0 - 1.0;
}

float quinticSmooth(float t) {
  float x = clamp(t, 0.0, 1.0);
  return x * x * x * (x * (x * 6.0  - 15.0) + 10.0);
}

float bias(float b, float t) {
  return pow(t, log(b) / log(0.5));
}

float smin(float a, float b, float k) {
  float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

float sdPlane(vec3 p)
{
	return p.y;
}

float sdEllipsoid(in vec3 p, in vec3 r)
{
    float k0 = length(p/r);
    float k1 = length(p/(r*r));
    return k0*(k0-1.0)/k1;
}

float dot2( in vec2 v ) {
  return dot(v,v);
}

float dot2(in vec3 v) {
  return dot(v,v);
}

float sdCappedCone( in vec3 p, in float h, in float r1, in float r2 )
{
  vec2 q = vec2( length(p.xz), p.y );

  vec2 k1 = vec2(r2,h);
  vec2 k2 = vec2(r2-r1,2.0*h);
  vec2 ca = vec2(q.x-min(q.x,(q.y < 0.0)?r1:r2), abs(q.y)-h);
  vec2 cb = q - k1 + k2*clamp( dot(k1-q,k2)/dot2(k2), 0.0, 1.0 );
  float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
  return s*sqrt( min(dot2(ca),dot2(cb)) );
}

float sdCylinder(vec3 p, vec3 a, vec3 b, float r)
{
    vec3 pa = p - a;
    vec3 ba = b - a;
    float baba = dot(ba,ba);
    float paba = dot(pa,ba);

    float x = length(pa*baba-ba*paba) - r*baba;
    float y = abs(paba-baba*0.5)-baba*0.5;
    float x2 = x*x;
    float y2 = y*y*baba;
    float d = (max(x,y)<0.0)?-min(x2,y2):(((x>0.0)?x2:0.0)+((y>0.0)?y2:0.0));
    return sign(d)*sqrt(abs(d))/baba;
}

float sdHexPrism(vec3 p, vec2 h)
{
    vec3 q = abs(p);

    const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
    p = abs(p);
    p.xy -= 2.0*min(dot(k.xy, p.xy), 0.0)*k.xy;
    vec2 d = vec2(
       length(p.xy - vec2(clamp(p.x, -k.z*h.x, k.z*h.x), h.x))*sign(p.y - h.x),
       p.z-h.y );
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdPlain(vec3 pos) {
  return sdPlane(pos - vec3(0.0, -110.0, 0.0));
}

float sdPupil(vec3 pos) {
  float seconds = u_Time / 1000.0;

  vec3 pupilPosition = vec3(0.0, 0.0, -1.25);
  vec4 i = vec4(((rotationMatrix(vec3(1.0, 0.0, 0.0), M_PI * (smoothstep(0.0, 1.0, sin(seconds * M_PI * 0.2) + 0.5) * 0.15 + 0.15)) * vec4(pos, 1.0)).xyz - pupilPosition), 1.0);
  vec3 inversePos = i.xyz;

	inversePos.xz = abs(inversePos.xz) + 0.625;
	float innerPupil = length(inversePos) - PUPIL_RADIUS;
  return max(length(inversePos) - PUPIL_RADIUS - 0.125, -innerPupil);
}

float sdPupilInside(vec3 pos) {
  float seconds = u_Time / 1000.0;

  vec3 pupilPosition = vec3(0.0, 0.0, -1.25);
  vec4 i = vec4(((rotationMatrix(vec3(1.0, 0.0, 0.0), M_PI * (smoothstep(0.0, 1.0, sin(seconds * M_PI * 0.2) + 0.5) * 0.15 + 0.15)) * vec4(pos, 1.0)).xyz - pupilPosition), 1.0);
  vec3 inversePos = i.xyz;

	inversePos.xz = abs(inversePos.xz) + 0.625;
  return length(inversePos) - PUPIL_RADIUS;
}

float sdEye(vec3 pos) {
  float seconds = u_Time / 1000.0;

  vec3 eyePosition = vec3(0.0, 0.0, 0.0);
  vec4 i = vec4(((rotationMatrix(vec3(1.0, 0.0, 0.0), M_PI * (smoothstep(0.0, 1.0, sin(seconds * M_PI * 0.2) + 0.5) * 0.15 + 0.15)) * vec4(pos, 1.0)).xyz - eyePosition), 1.0);
  vec3 inversePos = i.xyz;

  float ellipsoid = sdEllipsoid(inversePos, EYE_RADIUS * vec3(1.5, 0.9, 0.9));
  return ellipsoid;
}

float sdBaradDur(vec3 pos) {
  vec3 towerTop = vec3(0.0, 5.0, 0.0);
  vec3 spireBottom = vec3(0.0, -10.0, 0.0);
  vec3 towerBottom = vec3(0.0, -50.0, 0.0);
  vec3 secondBottom = vec3(0.0, -65.0, 0.0);
  vec3 thirdBottom = vec3(0.0, -80.0, 0.0);
  vec3 fourthBottom = vec3(0.0, -95.0, 0.0);
  vec3 fifthBottom = vec3(0.0, -110.0, 0.0);
  float mainTurretRadius = EYE_RADIUS * 2.0;

  float spireTurret = sdCylinder(pos, towerTop, spireBottom, mainTurretRadius);
  float mainTurret = sdCappedCone(pos - towerBottom, length(spireBottom - towerBottom), mainTurretRadius * 0.85, mainTurretRadius * 0.75);
  float spire1 = sdEllipsoid(pos + vec3(0.0, -5.0, EYE_RADIUS * 0.25), vec3(EYE_RADIUS * 2.0, EYE_RADIUS * 3.0, EYE_RADIUS * 5.0));
  float spire2 = sdEllipsoid(pos + vec3(0.0, -5.0, -EYE_RADIUS * 0.25), vec3(EYE_RADIUS * 2.0, EYE_RADIUS * 3.0, EYE_RADIUS * 5.0));
  float spire3 = sdEllipsoid(pos + vec3(0.0, -10.0, 0.0), vec3(EYE_RADIUS * 2.0, EYE_RADIUS * 3.0, EYE_RADIUS * 5.0));
  float spire = max(max(max(spireTurret, -spire1), -spire2), -spire3);

  float secondTurret = sdCappedCone(pos - secondBottom, length(towerBottom - secondBottom), mainTurretRadius * 1.25, mainTurretRadius * 0.85);
  float thirdTurret = sdCappedCone(pos - thirdBottom, length(secondBottom - thirdBottom), mainTurretRadius * 1.5, mainTurretRadius * 1.5);
  float fourthTurret = sdCappedCone(pos - fourthBottom, length(thirdBottom - fourthBottom), mainTurretRadius * 2.0, mainTurretRadius * 2.0);
  float fifthTurret = sdCappedCone(pos - fifthBottom, length(fourthBottom - fifthBottom), mainTurretRadius * 3.0, mainTurretRadius * 2.5);
  float mainTurrets = smin(smin(smin(smin(smin(spire, mainTurret, 0.9), secondTurret, 0.9), thirdTurret, 0.9), fourthTurret, 0.9), fifthTurret, 0.9);

  vec3 watch1 = (rotationMatrix(vec3(1.0, 0.0, 0.0), M_PI * 0.5) * vec4(vec3(pos - fifthBottom - vec3(0.0, 2.0, -mainTurretRadius * 2.75)), 1.0)).xyz;
  float watchTurret1 = sdHexPrism(watch1, vec2(2.0, 18.0));
  vec3 watch2 = (rotationMatrix(vec3(1.0, 0.0, 0.0), M_PI * 0.5) * vec4(vec3((rotationMatrix(vec3(0.0, 1.0, 0.0), M_PI * 0.125) * vec4(pos, 1.0)).xyz - fifthBottom - vec3(0.0, 2.0, -mainTurretRadius * 2.75)), 1.0)).xyz;
  float watchTurret2 = sdHexPrism(watch2, vec2(2.0, 18.0));
  vec3 watch3 = (rotationMatrix(vec3(1.0, 0.0, 0.0), M_PI * 0.5) * vec4(vec3(pos - fifthBottom - vec3(0.0, 2.0, mainTurretRadius * 2.75)), 1.0)).xyz;
  float watchTurret3 = sdHexPrism(watch3, vec2(2.0, 18.0));
  vec3 watch4 = (rotationMatrix(vec3(1.0, 0.0, 0.0), -M_PI * 0.5) * vec4(vec3((rotationMatrix(vec3(0.0, 1.0, 0.0), M_PI * 0.125) * vec4(pos, 1.0)).xyz - fifthBottom - vec3(0.0, 2.0, mainTurretRadius * 2.75)), 1.0)).xyz;
  float watchTurret4 = sdHexPrism(watch4, vec2(2.0, 18.0));
  vec3 watch5 = (rotationMatrix(vec3(1.0, 0.0, 0.0), M_PI * 0.5) * vec4(vec3((rotationMatrix(vec3(0.0, 1.0, 0.0), M_PI * 0.5625) * vec4(pos, 1.0)).xyz - fifthBottom - vec3(0.0, 2.0, -mainTurretRadius * 2.75)), 1.0)).xyz;
  float watchTurret5 = sdHexPrism(watch5, vec2(2.0, 18.0));
  vec3 watch6 = (rotationMatrix(vec3(1.0, 0.0, 0.0), M_PI * 0.5) * vec4(vec3((rotationMatrix(vec3(0.0, 1.0, 0.0), -M_PI * 0.4375) * vec4(pos, 1.0)).xyz - fifthBottom - vec3(0.0, 2.0, -mainTurretRadius * 2.75)), 1.0)).xyz;
  float watchTurret6 = sdHexPrism(watch6, vec2(2.0, 18.0));

  float lowTurrets = min(min(min(min(min(min(mainTurrets, watchTurret1), watchTurret2), watchTurret3), watchTurret4), watchTurret5), watchTurret6);

  for (int i = 0; i < 4; i++) {
    watch1 = (rotationMatrix(vec3(1.0, 0.0, 0.0), M_PI * 0.5) * vec4(vec3((rotationMatrix(vec3(0.0, 1.0, 0.0), M_PI * 0.5 * float(i) + M_PI * 0.375) * vec4(pos, 1.0)).xyz - fourthBottom - vec3(0.0, 2.5, -mainTurretRadius * 2.5)), 1.0)).xyz;
    watchTurret1 = sdHexPrism(watch1, vec2(2.0, 15.0));
    lowTurrets = min(lowTurrets, watchTurret1);
  }
  for (int i = 0; i < 2; i++) {
    watch1 = (rotationMatrix(vec3(1.0, 0.0, 0.0), M_PI * 0.5) * vec4(vec3((rotationMatrix(vec3(0.0, 1.0, 0.0), M_PI * float(i) + M_PI * 0.5625) * vec4(pos, 1.0)).xyz - thirdBottom - vec3(0.0, 1.5, -mainTurretRadius * 2.0)), 1.0)).xyz;
    watchTurret1 = sdHexPrism(watch1, vec2(2.0, 15.0));
    lowTurrets = min(lowTurrets, watchTurret1);
  }
  for (int i = 0; i < 2; i++) {
    watch1 = (rotationMatrix(vec3(1.0, 0.0, 0.0), M_PI * 0.5) * vec4(vec3((rotationMatrix(vec3(0.0, 1.0, 0.0), M_PI * float(i) + M_PI * 0.0625) * vec4(pos, 1.0)).xyz - secondBottom - vec3(0.0, 1.0, -mainTurretRadius * 1.5)), 1.0)).xyz;
    watchTurret1 = sdHexPrism(watch1, vec2(2.0, 15.0));
    lowTurrets = min(lowTurrets, watchTurret1);
  }
  return lowTurrets;
}

vec4 rayMarch(vec3 rayDirec, inout int object, inout float closestDistToEye, inout vec3 closestPointToEye) {
  float depth = 0.0;
  for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
    vec3 current = u_Eye + depth * rayDirec;
    float eyeDist = sdEye(current);
    float pupilDist = sdPupil(current);
    float pupilInsideDist = sdPupilInside(current);
    float baradDurDist = sdBaradDur(current);
    float plainDist = sdPlain(current);
    if (eyeDist < closestDistToEye) {
      closestDistToEye = eyeDist;
      closestPointToEye = current;
    }
    pupilDist = max(eyeDist, pupilDist);
    float dist = min(min(min(max(eyeDist, -pupilInsideDist), pupilDist), baradDurDist), plainDist);
    if (dist < EPSILON) {
      if (abs(pupilDist - dist) < EPSILON) {
        object = 2;
        return vec4(current, depth);
      }
      if (abs(eyeDist - dist) < EPSILON) {
        object = 1;
        return vec4(current, depth);
      }
      if (abs(baradDurDist - dist) < EPSILON) {
        object = 3;
        return vec4(current, depth);
      }
      if (abs(plainDist - dist) < EPSILON) {
        object = 4;
        return vec4(current, depth);
      }
    }
    depth += dist;
    if (depth >= MAX_DIST) {
      object = 0;
      return vec4(current, depth);
    }
  }
  object = -1;
  return vec4(vec3(u_Eye + depth * rayDirec), depth);
}

float worleyNoise(vec2 pos) {
  float factor = 8.0;
  vec2 seed = vec2(0.0, 0.0);

  int x = int(floor(pos.x / factor));
  int y = int(floor(pos.y / factor));
  vec2 minWorley = factor * randvec3(vec2(float(x), float(y)), seed) + vec2(float(x) * factor, float(y) * factor);
  float minDist = distance(minWorley, pos);
  for (int i = x - 1; i <= x + 1; i++) {
      for (int j = y - 1; j <= y + 1; j++) {
          vec2 worley = factor * randvec3(vec2(float(i), float(j)), seed) + vec2(float(i) * factor, float(j) * factor);
          if (minDist > distance(pos, worley)) {
              minDist = distance(pos, worley);
              minWorley = worley;
          }
      }
  }
  return clamp(minDist / (factor * 2.0), 0.0, 0.5);
}

float interpRand(float x, float z) {
  vec2 seed = vec2(0.0, 0.0);

  float intX = floor(x);
  float fractX = fract(x);
  float intZ = floor(z);
  float fractZ = fract(z);

  vec2 c1 = vec2(intX, intZ);
  vec2 c2 = vec2(intX + 1.0, intZ);
  vec2 c3 = vec2(intX, intZ + 1.0);
  vec2 c4 = vec2(intX + 1.0, intZ + 1.0);

  float v1 = random1(c1, seed);
  float v2 = random1(c2, seed);
  float v3 = random1(c3, seed);
  float v4 = random1(c4, seed);

  float i1 = mix(v1, v2, quinticSmooth(fractX));
  float i2 = mix(v3, v4, quinticSmooth(fractX));
  return mix(i1, i2, quinticSmooth(fractZ));
}

float eyeTexture(vec3 pos) {
  float seconds = u_Time / 1000.0;
  pos = (rotationMatrix(vec3(1.0, 0.0, 0.0), M_PI * (smoothstep(0.0, 1.0, sin(seconds * M_PI * 0.2) + 0.5) * 0.15 + 0.15)) * vec4(pos, 1.0)).xyz;
  float distFromSurface = max(0.0, 3.0 - (3.0 * length(pos)));
  vec3 coord = vec3(atan(pos.x, pos.y * 1.5) / (2.0 * M_PI) + 0.5, length(pos) * 0.4, 0.5 + pos.z);
	for (int i = 1; i <= 7; i++) {
		float power = pow(2.0, float(i));
		distFromSurface += (1.5 / power) * fireNoise(coord + vec3(0.0, -seconds * 0.05, -seconds * 0.01), power * 16.0);
	}
  return distFromSurface;
}

float baradDurTexture(float x, float y, float z) {
  float total = 0.0;
  int octaves = 16;
  float persistence = 0.7;
  for (int i = 0; i < octaves; i++) {
    float freq = pow(2.0, float(i));
    float amp = pow(persistence, float(i));
    total += worleyNoise(vec2(x + z * freq, y * freq)) * amp;
  }
  return total;
}

float skyTexture(vec3 pos) {
  float total = 0.0;
  int octaves = 8;
  float persistence = 0.4;
  for (int i = 0; i < octaves; i++) {
    float freq = pow(2.0, float(i)) * 0.0001;
    float amp = pow(persistence, float(i));
    total += interpRand(pos.x * freq, pos.z * freq) * amp;
  }
  if (total < 0.75) {
    return 1.0 - bias(0.3, total);
  }
  return 1.0 - bias(0.3, total);
}

float findSDGivenPoint(vec3 current) {
  float eyeDist = sdEye(current);
  float pupilDist = sdPupil(current);
  float pupilInsideDist = sdPupilInside(current);
  float baradDurDist = sdBaradDur(current);
  float plainDist = sdPlain(current);
  pupilDist = max(eyeDist, pupilDist);
  return min(min(min(max(eyeDist, -pupilInsideDist), pupilDist), baradDurDist), plainDist);
}

float findSDGivenPointNoPlain(vec3 current) {
  float eyeDist = sdEye(current);
  float pupilDist = sdPupil(current);
  float pupilInsideDist = sdPupilInside(current);
  float baradDurDist = sdBaradDur(current);
  pupilDist = max(eyeDist, pupilDist);
  return min(min(max(eyeDist, -pupilInsideDist), pupilDist), baradDurDist);
}

vec3 estimateNormal(vec3 p) {
  float x1 = findSDGivenPoint(vec3(p.x + NORMAL_EPSILON, p.y, p.z));
  float x2 = findSDGivenPoint(vec3(p.x - NORMAL_EPSILON, p.y, p.z));
  float y1 = findSDGivenPoint(vec3(p.x, p.y + NORMAL_EPSILON, p.z));
  float y2 = findSDGivenPoint(vec3(p.x, p.y - NORMAL_EPSILON, p.z));
  float z1 = findSDGivenPoint(vec3(p.x, p.y, p.z + NORMAL_EPSILON));
  float z2 = findSDGivenPoint(vec3(p.x, p.y, p.z - NORMAL_EPSILON));
  return normalize(vec3(x1 - x2, y1 - y2, z1 - z2));
}

float softShadow(vec3 dir, vec3 origin, float k) {
  float res = 1.0;
  for (float t = 0.1; t < MAX_DIST; ) {
    float m = findSDGivenPointNoPlain(origin + t * dir);
    if (m < EPSILON) {
      return 0.0;
    }
    res = min(res, k * m / t);
    t += m;
  }
  return res;
}

vec3 applyFog(in vec3 rgb, in float dist, in vec3 rayDir, in vec3 sunDir, in float b)
{
    float fogAmount = 1.0 - exp(-dist * b);
    vec3 fogColor = vec3(0.0625,0.0625,0.0625);
    return mix(rgb, fogColor, fogAmount);
}

void main() {
  float fovy = 90.0;
  vec3 look = u_Ref - u_Eye;
  vec3 right = normalize(cross(look, u_Up));
  float aspect = float(u_Dimensions.x) / float(u_Dimensions.y);
  float tan_fovy2 = tan(fovy / 2.0);
  vec3 h = right * length(look) * aspect * tan_fovy2;
  vec3 v = u_Up * length(look) * tan_fovy2;
  vec3 p = u_Ref + fs_Pos.x * h + fs_Pos.y * v;
  vec3 rayDirect = normalize(p - u_Eye);

  float seconds = u_Time / 1000.0;

  out_Col = vec4(0.0, 0.0, 0.0, 1.0);

  bool skyEdited = false;
  int objectHit = 0;
  float closestDistToEye = MAX_DIST;
  vec3 closestPointToEye = u_Eye;
  bool noBackground = false;
  vec4 target = rayMarch(rayDirect, objectHit, closestDistToEye, closestPointToEye);
  if (objectHit == 1) {
    float dist = eyeTexture(target.xyz / (EYE_RADIUS * 1.5));
    vec3 rotatedTarget = (rotationMatrix(vec3(1.0, 0.0, 0.0), M_PI * (smoothstep(0.0, 1.0, sin(seconds * M_PI * 0.2) + 0.5) * 0.15 + 0.15)) * vec4(target.xyz, 1.0)).xyz;
    float centerMix = clamp(bias(0.1, length(vec2(rotatedTarget.x * 3.0, rotatedTarget.y * 1.5)) / EYE_RADIUS), 0.0, 1.0);
    out_Col = mix(vec4(1.0, 1.0, 0.973, 1.0), vec4(dist, pow(max(dist, 0.0), 2.0) * 0.4, pow(max(dist, 0.0), 3.0) * 0.15, 1.0), clamp(centerMix, 0.0, 1.0));
  } else if (objectHit == 2) {
    if (abs(closestDistToEye) < 2.0 * EPSILON) {
      out_Col = vec4(1.0, 1.0, 1.0, 1.0);
    } else {
      out_Col = vec4(0.0, 0.0, 0.0, 1.0);
      noBackground = true;
    }
  } else if (objectHit == 3) {
    float baradDurTexture = baradDurTexture(target.x, target.y, target.z);
    out_Col = vec4(vec3(0.1, 0.1, 0.1) * baradDurTexture, 1.0);
  } else if (objectHit == 4) {
    out_Col = vec4(0.1, 0.1, 0.1, 1.0);
  }
  if (!noBackground && objectHit <= 2) {
    float dist = eyeTexture(p.xyz / (EYE_RADIUS * 1.5));
    out_Col = mix(out_Col, vec4(dist * 1.2, pow(max(dist, 0.0), 2.0) * 0.4 * 1.2, pow(max(dist, 0.0), 3.0) * 0.15 * 1.2, 1.0), smoothstep(0.0, 1.0, clamp(length(vec2(p.x, p.y)) / (EYE_RADIUS * 1.5), 0.0, 1.0)));

    if (rayDirect.y > 0.0) {
      float skyDist = (MAX_DIST * 10.0 - u_Eye.y) / rayDirect.y;
      float cloudCover = skyTexture(u_Eye + rayDirect * skyDist);
      float lightningTime = mod(seconds, 3.0);
      float lightning = smoothstep(0.4, 0.0, lightningTime) * 1.5 + smoothstep(0.4, 0.0, (lightningTime < 0.8) ? 1.0 : lightningTime - 0.8) * 2.5 + smoothstep(0.4, 0.0,  (lightningTime < 1.0) ? 1.0 : lightningTime - 1.0) * 1.0;
      out_Col = mix(out_Col, vec4(vec3(0.2, 0.2, 0.3) * cloudCover + lightning * 5.0 * smoothstep(0.0, 1.0, cloudCover) * vec3(0.5, 0.5, 0.5), 1.0), clamp(closestDistToEye, 0.0, 1.0));
      skyEdited = true;
    } else {
      out_Col = mix(out_Col, vec4(0.0, 0.0, 0.0, 1.0), clamp(closestDistToEye, 0.0, 1.0));
    }
  }

  vec3 eyePos = vec3(0.0, 0.0, 0.0);
  vec3 sunPos = vec3(0.0, 50.0, -500.0);
  vec3 sunVec = sunPos - target.xyz;

  if (objectHit > 2) {
    vec3 lightVec = eyePos - target.xyz;
    vec3 normal = estimateNormal(target.xyz);
    float diffuseTerm = dot(normalize(normal), normalize(lightVec));
    diffuseTerm = clamp(diffuseTerm, 0.0, 1.0);
    float ambientTerm = 0.6;
    vec3 intensities = vec3(88.6, 34.5, 13.3);

    vec3 cameraPosition = eyePos;
    vec3 view = normalize(target.xyz - cameraPosition);
    vec3 light = normalize(target.xyz - eyePos);
    float specularIntensity = dot(reflect(-light, normal), view);
    specularIntensity = pow(max(specularIntensity, 0.0), 2.0);

    float attenuationFactor = 0.005;
    vec3 beforeSpecular = vec3(out_Col.x * intensities.x, out_Col.y * intensities.y, out_Col.z * intensities.z);
    float attenuation = 1.0 / (1.0 + attenuationFactor * pow(length(lightVec), 2.0));
    out_Col = vec4(ambientTerm * out_Col.xyz + beforeSpecular * attenuation * (diffuseTerm * out_Col.xyz + vec3(0.886, 0.345, 0.133) * specularIntensity), 1.0);

    float coneAngle = 5.0;
    attenuationFactor = 0.0;
    attenuation = 1.0 / (1.0 + attenuationFactor * pow(length(lightVec), 2.0));
    intensities = vec3(88.6, 34.5, 13.3) * 10.0;
    beforeSpecular = vec3(out_Col.x * intensities.x, out_Col.y * intensities.y, out_Col.z * intensities.z);
    vec3 coneDirection = (rotationMatrix(vec3(1.0, 0.0, 0.0), M_PI - M_PI * (smoothstep(0.0, 1.0, sin(seconds * M_PI * 0.2) + 0.5) * 0.15 + 0.15)) * vec4(0.0, 0.0, 1.0, 1.0)).xyz;
    float lightToSurfaceAngle = degrees(acos(dot(normalize(-lightVec), normalize(coneDirection))));
    if (lightToSurfaceAngle <= coneAngle) {
      if (objectHit != 3) {
        out_Col += vec4(beforeSpecular * attenuation * diffuseTerm * out_Col.xyz * 0.5, 0.0);
      }
    }

    float sunDiffuseTerm = dot(normalize(normal), normalize(sunVec));
    sunDiffuseTerm = clamp(sunDiffuseTerm, 0.0, 1.0);
    ambientTerm = 0.2;
    float sunIntensity = sunDiffuseTerm * 10.0 + ambientTerm;
    float shadow = softShadow(normalize(-sunVec), target.xyz, 1.0);
    out_Col = vec4(out_Col.xyz + out_Col.xyz * sunIntensity * shadow, 1000.0);
  }
  if (objectHit <= 0) {
    vec3 endPoint = u_Eye + 1.2 * MAX_DIST * rayDirect;
    out_Col = vec4(applyFog(out_Col.xyz, length(vec2(endPoint.x - u_Eye.x, endPoint.z - u_Eye.z)), rayDirect, normalize(u_Eye - sunPos), 0.004), 1.0);
  } else {
    out_Col = vec4(applyFog(out_Col.xyz, length(vec2(target.x - u_Eye.x, target.z - u_Eye.z)), rayDirect, normalize(u_Eye - sunPos), 0.0045), 1.0);
  }
}
