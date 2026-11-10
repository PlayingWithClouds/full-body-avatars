/**
 * The studio's light, for what the viewer shades itself (hair, eyes). The skin's lighting is
 * baked into its textures from the videos; this light comes from about where theirs did.
 */

import * as THREE from 'three';

export interface Lighting {
	/** Unit direction towards the key light, its colour; the same for the fill light. */
	key: THREE.Vector3;
	keyColour: THREE.Color;
	fill: THREE.Vector3;
	fillColour: THREE.Color;
	ambient: THREE.Color;
}

/** Fixed in the world: key from the front, above and to the avatar's left. */
export const LIGHTING: Lighting = {
	key: new THREE.Vector3(0.45, -0.6, 0.65).normalize(),
	keyColour: new THREE.Color(3.2, 3.05, 2.9),
	fill: new THREE.Vector3(-0.6, 0.35, 0.25).normalize(),
	fillColour: new THREE.Color(0.95, 0.95, 0.95),
	ambient: new THREE.Color(0.55, 0.55, 0.55)
};
