/*
* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* ============================================================================
* Copyright 2021 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
//
//
//		0==========================0
//		|    Local feature test    |
//		0==========================0
//
//		version 1.0 : 
//			> 
//
//---------------------------------------------------
//
//		Cloud source :
//		Define usefull Functions/Methods
//
//----------------------------------------------------
//
//		Hugues THOMAS - 10/02/2017
//


#include "cloud.h"


// Getters
// *******

PointXYZ max_point(std::vector<PointXYZ> points)
{
	// Initiate limits
	PointXYZ maxP(points[0]);

	// Loop over all points
	for (auto p : points)
	{
		if (p.x > maxP.x)
			maxP.x = p.x;

		if (p.y > maxP.y)
			maxP.y = p.y;

		if (p.z > maxP.z)
			maxP.z = p.z;
	}

	return maxP;
}

PointXYZ min_point(std::vector<PointXYZ> points)
{
	// Initiate limits
	PointXYZ minP(points[0]);

	// Loop over all points
	for (auto p : points)
	{
		if (p.x < minP.x)
			minP.x = p.x;

		if (p.y < minP.y)
			minP.y = p.y;

		if (p.z < minP.z)
			minP.z = p.z;
	}

	return minP;
}