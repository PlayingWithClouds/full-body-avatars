/**
 * Hair shading in GLSL.
 *
 * The fibre scatters light the way Marschner et al. (2003) measured: a white reflection off the
 * cuticle (R) shifted towards the root, light transmitted through the fibre (TT), strong when
 * the hair is lit from behind, and light reflected once inside it (TRT), the coloured second
 * highlight. The lobes use the closed-form approximations of Karis (2016, "Physically Based Hair
 * Shading in Unreal"): Gaussian longitudinal terms and fitted azimuthal ones. Colour comes from
 * melanin, as absorption per fibre diameter (d'Eon et al. 2011; Chiang et al. 2016), so the
 * same pigment darkens TT and TRT by the path the light takes through the fibre.
 *
 * Shadows within the hair come from a deep opacity map (Yuksel and Keyser 2008): from the
 * light, the depth of the first hair per texel and the hair's opacity accumulated in four
 * layers behind it. Light that passes through hair keeps some of its colour, as it does through
 * blonde hair, rather than going black.
 */

/** Depth, in metres behind the first hair, of the ends of the deep opacity map's four layers. */
export const OPACITY_LAYERS = [0.003, 0.01, 0.025, 0.07];

/** Absorption per fibre diameter of eumelanin and pheomelanin concentrations. */
export const MELANIN_GLSL = /* glsl */ `
vec3 melaninAbsorption(float eumelanin, float pheomelanin) {
	return eumelanin * vec3(0.419, 0.697, 1.37) + pheomelanin * vec3(0.187, 0.4, 1.05);
}
`;

export const SHADING_GLSL = /* glsl */ `
const float PI = 3.14159265;

float longitudinal(float width, float x) {
	return exp(-0.5 * x * x / (width * width)) / (sqrt(2.0 * PI) * width);
}

float fresnel(float cosine) {
	const float F0 = 0.0465; // ((1 - 1.55) / (1 + 1.55))^2
	return F0 + (1.0 - F0) * pow(1.0 - cosine, 5.0);
}

/**
 * Light scattered towards unit view direction V from unit light direction L by a fibre with unit
 * tangent T, absorption sigma per diameter and roughness (0 to 1), per unit irradiance.
 */
vec3 hairScattering(vec3 T, vec3 V, vec3 L, vec3 sigma, float roughness) {
	float sinL = clamp(dot(T, L), -1.0, 1.0);
	float sinV = clamp(dot(T, V), -1.0, 1.0);
	float cosD = cos(0.5 * abs(asin(sinV) - asin(sinL)));
	vec3 Lp = L - sinL * T;
	vec3 Vp = V - sinV * T;
	float cosPhi = dot(Lp, Vp) * inversesqrt(dot(Lp, Lp) * dot(Vp, Vp) + 1e-4);
	float cosHalfPhi = sqrt(clamp(0.5 + 0.5 * cosPhi, 0.0, 1.0));
	// Effective index of refraction in the fibre's cross-section (Karis's fit of Bravais's).
	float n = 1.19 / cosD + 0.36 * cosD;
	float a = 1.0 / n;
	// Cuticle scales tilt the lobes: R towards the root, TT and TRT towards the tip.
	const float SHIFT = 0.04;
	float beta = max(roughness * roughness, 0.02);

	// R: reflection off the surface.
	float mR = longitudinal(beta * sqrt(2.0) * max(cosHalfPhi, 0.2), sinL + sinV + 2.0 * SHIFT);
	vec3 r = vec3(mR * 0.25 * cosHalfPhi * fresnel(sqrt(clamp(0.5 + 0.5 * dot(V, L), 0.0, 1.0))));

	// TT: through the fibre, forward.
	float mTT = longitudinal(beta * 0.5, sinL + sinV - SHIFT);
	float h = cosHalfPhi * (1.0 + a * (0.6 - 0.8 * cosPhi));
	float fTT = fresnel(cosD * sqrt(clamp(1.0 - h * h, 0.0, 1.0)));
	vec3 absorbedTT = exp(-sigma * 2.0 * sqrt(clamp(1.0 - h * h * a * a, 0.0, 1.0)) / cosD);
	vec3 tt = mTT * exp(-3.65 * cosPhi - 3.98) * (1.0 - fTT) * (1.0 - fTT) * absorbedTT;

	// TRT: reflected once inside, back towards the light.
	float mTRT = longitudinal(beta * 2.0, sinL + sinV - 4.0 * SHIFT);
	float fTRT = fresnel(cosD * 0.5);
	vec3 absorbedTRT = exp(-sigma * 4.0 * sqrt(1.0 - 0.75 * a * a) / cosD);
	vec3 trt = mTRT * exp(17.0 * cosPhi - 16.78) * (1.0 - fTRT) * (1.0 - fTRT) * fTRT * absorbedTRT;

	return (r + tt + trt) * sqrt(1.0 - sinL * sinL);
}

/**
 * Light scattered many times between fibres, the soft glow that makes light hair look light: a
 * wrapped diffuse term tinted by the fibres' colour, deeper where the light came through hair.
 * It passes through at least two fibres, so dark hair has next to none of it.
 */
vec3 multipleScattering(vec3 T, vec3 V, vec3 L, vec3 sigma, vec3 transmittance) {
	vec3 albedo = exp(-2.0 * sigma);
	vec3 facing = normalize(V - T * dot(V, T));
	float wrapped = clamp((dot(facing, L) + 1.0) * 0.25, 0.0, 1.0);
	float kajiya = 1.0 - abs(dot(T, L));
	float shade = mix(wrapped, kajiya, 0.33) / PI;
	float through = dot(transmittance, vec3(0.3333));
	vec3 tint = pow(albedo / max(dot(albedo, vec3(0.3333)), 1e-3), vec3(1.0 - through));
	return albedo * shade * tint;
}
`;

/**
 * GLSL to read a deep opacity map: `opacityDepth` (the light's depth of the first hair),
 * `opacityLayers` (the hair's optical depth in the four layers behind it, cumulative), the
 * light's view-projection and its depth range in metres.
 */
export function opacityMapGlsl(suffix = ''): string {
	return /* glsl */ `
uniform sampler2D opacityDepth${suffix};
uniform sampler2D opacityLayers${suffix};
uniform mat4 lightMatrix${suffix};
uniform float lightRange${suffix};

/**
 * Optical depth of the hair between the light and world point p, averaged over a disc of
 * \`spread\` texels with the layers prefiltered over about 2^lod: single strands are a texel
 * wide in the map, and their shadows would be dots.
 */
float opticalDepth${suffix}(vec3 p, float spread, float lod) {
	vec4 clip = lightMatrix${suffix} * vec4(p, 1.0);
	vec3 ndc = clip.xyz / clip.w;
	vec2 uv = ndc.xy * 0.5 + 0.5;
	if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return 0.0;
	float depth = ndc.z * 0.5 + 0.5;
	vec2 texel = spread / vec2(textureSize(opacityDepth${suffix}, 0));
	float total = 0.0;
	for (int k = 0; k < 8; k++) {
		float angle = float(k) * 2.3999632 + 0.4;
		vec2 offset = sqrt((float(k) + 0.5) / 8.0) * vec2(cos(angle), sin(angle)) * texel;
		float front = texture(opacityDepth${suffix}, uv + offset).r;
		// No hair at this texel: the layers, blended from its neighbours, tell how much.
		float behind = front >= 1.0 ? 1.0 : (depth - front) * lightRange${suffix} - 0.0015;
		vec4 layers = textureLod(opacityLayers${suffix}, uv + offset, lod);
		const vec4 ENDS = vec4(${OPACITY_LAYERS.join(', ')});
		float sum;
		if (behind <= 0.0) sum = 0.0;
		else if (behind < ENDS.x) sum = layers.x * behind / ENDS.x;
		else if (behind < ENDS.y) sum = mix(layers.x, layers.y, (behind - ENDS.x) / (ENDS.y - ENDS.x));
		else if (behind < ENDS.z) sum = mix(layers.y, layers.z, (behind - ENDS.y) / (ENDS.z - ENDS.y));
		else if (behind < ENDS.w) sum = mix(layers.z, layers.w, (behind - ENDS.z) / (ENDS.w - ENDS.z));
		else sum = layers.w;
		total += sum;
	}
	return total / 8.0;
}

`;
}
